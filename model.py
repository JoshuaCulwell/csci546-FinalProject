import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def timestep_embedding(t, dim):
    """Sinusoidal timestep embedding."""
    device = t.device
    half = dim // 2

    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=device) / half * 10
    )

    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# ------------------------------------------------------------
# Conditional Diffusion Transition Model
# ------------------------------------------------------------

class DiffusionTransition(nn.Module):
    """
    εθ(h_noisy, h_prev, token_embed, t)
    Predicts noise for denoising diffusion.
    """
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.fc_h = nn.Linear(dim, hidden)
        self.fc_prev = nn.Linear(dim, hidden)
        self.fc_tok = nn.Linear(dim, hidden)
        self.fc_t = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, h_noisy, h_prev, tok_embed, t):
        t_emb = timestep_embedding(t, h_noisy.size(-1))
        t_emb = self.fc_t(t_emb)

        x = self.fc_h(h_noisy) + self.fc_prev(h_prev) + self.fc_tok(tok_embed) + t_emb
        x = F.silu(x)
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

        # simple beta schedule
        betas = torch.linspace(1e-4, 0.02, num_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_cumprod", torch.cumprod(1 - betas, dim=0))

    # --------------------------------------------------------
    # Sample next hidden state via reverse diffusion
    # --------------------------------------------------------
    def sample_transition(self, h_prev, tok_embed):
        device = h_prev.device
        h = torch.randn_like(h_prev)

        for k in reversed(range(self.num_steps)):
            t = torch.full((h.size(0),), k, device=device, dtype=torch.long)
            eps = self.transition(h, h_prev, tok_embed, t)

            alpha = self.alphas[k]
            beta = self.betas[k]

            h = (h - beta * eps) / torch.sqrt(alpha)
            if k > 0:
                h += torch.sqrt(beta) * torch.randn_like(h)

        return h

    # --------------------------------------------------------
    # Diffusion loss for training
    # --------------------------------------------------------
    def diffusion_loss(self, h_target, h_prev, tok_embed):
        B = h_target.size(0)
        device = h_target.device

        t = torch.randint(0, self.num_steps, (B,), device=device)
        noise = torch.randn_like(h_target)

        alpha_bar = self.alpha_cumprod[t].view(B, 1)
        h_noisy = torch.sqrt(alpha_bar) * h_target + torch.sqrt(1 - alpha_bar) * noise

        pred_noise = self.transition(h_noisy, h_prev, tok_embed, t)
        return F.mse_loss(pred_noise, noise)

    # --------------------------------------------------------
    # Forward pass over a sequence
    # --------------------------------------------------------
    def forward(self, x):
        """
        x: (B, T) token indices
        """
        B, T = x.size()
        device = x.device

        h = torch.randn(B, self.dim, device=device)
        logits = []
        diff_losses = []

        for t in range(T):
            tok_embed = self.embed(x[:, t])

            # sample next hidden state
            h_next = self.sample_transition(h, tok_embed)

            # LM prediction
            logits.append(self.decoder(h_next))

            # diffusion loss (teacher forcing: target = h_next.detach())
            diff_losses.append(self.diffusion_loss(h_next.detach(), h, tok_embed))

            h = h_next

        logits = torch.stack(logits, dim=1)  # (B, T, vocab)
        diff_loss = torch.stack(diff_losses).mean()

        return logits, diff_loss

