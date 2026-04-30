import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def build_timestep_embedding(num_steps, dim, device):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=device) / half * 10)

    t = torch.arange(num_steps, device=device).float()
    args = t[:, None] * freqs[None]

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb  # (num_steps, dim)

# ------------------------------------------------------------
# Conditional Diffusion Transition Model (fused)
# ------------------------------------------------------------

class DiffusionTransition(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.in_proj = nn.Linear(dim * 4, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, h_noisy, h_prev, tok_embed, t_emb):
        x = torch.cat([h_noisy, h_prev, tok_embed, t_emb], dim=-1)
        x = F.silu(self.in_proj(x))
        return self.out(x)

# ------------------------------------------------------------
# Main Model
# ------------------------------------------------------------

class DiffusionRNN(nn.Module):
    def __init__(self, vocab_size, dim, num_steps=8):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps

        self.embed = nn.Embedding(vocab_size, dim)
        self.transition = DiffusionTransition(dim)
        self.decoder = nn.Linear(dim, vocab_size)

        # schedules
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

        # precompute timestep embeddings
        self.register_buffer(
            "timestep_emb",
            build_timestep_embedding(num_steps, dim, device=betas.device)
        )

    # --------------------------------------------------------
    # Reverse diffusion (optimized)
    # --------------------------------------------------------
    def sample_transition(self, h_prev, tok_embed):
        B = h_prev.size(0)
        device = h_prev.device

        h = torch.randn_like(h_prev)
        noise = torch.randn(self.num_steps, *h.shape, device=device)

        for k in range(self.num_steps - 1, -1, -1):
            t_emb = self.timestep_emb[k].expand(B, -1)

            eps = self.transition(h, h_prev, tok_embed, t_emb)

            alpha = self.alphas[k]
            beta = self.betas[k]

            h = (h - beta * eps) / torch.sqrt(alpha)

            if k > 0:
                h = h + torch.sqrt(beta) * noise[k]

        return h

    # --------------------------------------------------------
    # Diffusion loss
    # --------------------------------------------------------
    def diffusion_loss(self, h_target, h_prev, tok_embed):
        B = h_target.size(0)
        device = h_target.device

        t = torch.randint(0, self.num_steps, (B,), device=device)
        noise = torch.randn_like(h_target)

        alpha_bar = self.alpha_cumprod[t].unsqueeze(-1)
        h_noisy = torch.sqrt(alpha_bar) * h_target + torch.sqrt(1 - alpha_bar) * noise

        t_emb = self.timestep_emb[t]

        pred_noise = self.transition(h_noisy, h_prev, tok_embed, t_emb)
        return F.mse_loss(pred_noise, noise)

    # --------------------------------------------------------
    # Forward (less Python overhead)
    # --------------------------------------------------------
    def forward(self, x):
        B, T = x.size()
        device = x.device

        h = torch.randn(B, self.dim, device=device)

        logits = torch.empty(B, T, self.decoder.out_features, device=device)
        diff_losses = torch.empty(T, device=device)

        for t in range(T):
            tok_embed = self.embed(x[:, t])

            h_next = self.sample_transition(h, tok_embed)

            logits[:, t] = self.decoder(h_next)
            diff_losses[t] = self.diffusion_loss(h_next.detach(), h, tok_embed)

            h = h_next

        return logits, diff_losses.mean()
