import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    # Repeat to match full head dimension
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to query and key tensors."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # Ensure cos and sin have the right shape for broadcasting
    # q, k shape: (batch, n_head, seq_len, head_dim)
    # cos, sin shape should be: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout, bias):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        
        # key, query, value projections
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.size()

        # calculate query, key, values
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # apply rotary position embedding
        q, k = apply_rotary_pos_emb(q, k, freqs_cos[:T], freqs_sin[:T])

        # causal self-attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        hidden_dim = int(n_embd * 8 / 3)  # SwiGLU hidden dimension
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # Round to multiple of 64
        
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))

class Block(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout, bias):
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.post_attention_layernorm = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x, freqs_cos, freqs_sin):
        # Pre-norm attention
        x = x + self.attn(self.input_layernorm(x), freqs_cos, freqs_sin)
        # Pre-norm MLP
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=32000, n_layer=12, n_head=12, n_embd=768, block_size=1024, dropout=0.1, bias=True):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Embedding layers
        self.embed_tokens = nn.Embedding(vocab_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            Block(vocab_size, n_layer, n_head, n_embd, block_size, dropout, bias) 
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.embed_tokens.weight = self.lm_head.weight
        
        # Precompute rotary embeddings - use head_dim, not n_embd
        head_dim = n_embd // n_head
        freqs_cos, freqs_sin = precompute_freqs_cis(
            head_dim, block_size, theta=10000.0
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('down_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        print(f"Model initialized. Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Don't count embedding parameters twice due to weight tying
            n_params -= self.embed_tokens.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # Token embeddings
        x = self.dropout(self.embed_tokens(idx))

        # Forward through transformer blocks
        for layer in self.layers:
            x = layer(x, self.freqs_cos, self.freqs_sin)
        
        x = self.norm(x)

        if targets is not None:
            # Training mode: compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        N = self.get_num_params()
        L, H, Q, T = self.n_layer, self.n_head, self.n_embd//self.n_head, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    config = {
        'vocab_size': 32000,
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 1024,
        'dropout': 0.1,
        'bias': True 
    }

    model = GPT(**config)
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params/1e6:.2f}M")

    dummy_input = torch.randint(0, config['vocab_size'], (1, 10))
    logits, loss = model(dummy_input)
    print("Output logits shape:", logits.shape)