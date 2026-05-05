import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(num_steps, dim, device):
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=device) / half * 10
    )
    t = torch.arange(num_steps, device=device).float()
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class DiffusionTransition(nn.Module):
    def __init__(self, io_dim, hidden_dim):
        super().__init__()
        self.in_projection = nn.Linear(io_dim * 4, hidden_dim)
        self.out = nn.Linear(hidden_dim, io_dim)

    def forward(self, hidden_noisy, hidden_prev, token_embedding, time_embedding):
        out = torch.cat(
            [hidden_noisy, hidden_prev, token_embedding, time_embedding],
            dim=-1
        )
        out = F.silu(self.in_projection(out))
        return self.out(out)


class DiffusionRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_diff_steps,
        device,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_diff_steps = int(num_diff_steps)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transition = DiffusionTransition(embedding_dim, hidden_dim)
        self.decoder = nn.Linear(embedding_dim, vocab_size)

        betas = torch.linspace(1e-4, 0.02, self.num_diff_steps, device=device)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        timestep_embeddings = timestep_embedding(
            self.num_diff_steps,
            embedding_dim,
            device=device
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("timestep_embeddings", timestep_embeddings)

        self.hidden = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self):
        self.hidden = None

    def set_state(self, h):
        self.hidden = h

    def get_state(self):
        return self.hidden

    # ------------------------------------------------------------------
    # Diffusion core
    # ------------------------------------------------------------------

    def sample_transition(self, hidden_prev, token_embedding):
        B = hidden_prev.size(0)
        device = hidden_prev.device

        steps = self.num_diff_steps

        new_hidden = torch.randn_like(hidden_prev)
        noise = torch.randn(steps, B, hidden_prev.size(1), device=device)

        for k in range(steps - 1, -1, -1):
            t = self.timestep_embeddings[k].unsqueeze(0).expand(B, -1)

            predicted_noise = self.transition(
                new_hidden,
                hidden_prev,
                token_embedding,
                t
            )

            alpha = self.alphas[k]
            beta = self.betas[k]

            new_hidden = (new_hidden - beta * predicted_noise) / torch.sqrt(alpha)

            if k > 0:
                new_hidden = new_hidden + torch.sqrt(beta) * noise[k]

        return new_hidden

    def diffusion_loss(self, hidden_target, hidden_prev, token_embedding):
        B = hidden_target.size(0)
        device = hidden_target.device

        steps = self.num_diff_steps

        t = torch.randint(0, steps, (B,), device=device)
        noise = torch.randn_like(hidden_target)

        alpha_bar = self.alpha_cumprod[t].view(B, 1)

        hidden_noisy = torch.sqrt(alpha_bar) * hidden_target + \
                       torch.sqrt(1.0 - alpha_bar) * noise

        predicted_noise = self.transition(
            hidden_noisy,
            hidden_prev,
            token_embedding,
            self.timestep_embeddings[t]
        )

        return F.mse_loss(predicted_noise, noise)

    # ------------------------------------------------------------------
    # Step (for generation)
    # ------------------------------------------------------------------

    def step(self, x_t):
        token_embedding = self.embedding(x_t)

        if self.hidden is None:
            B = x_t.size(0)
            self.hidden = torch.randn(B, self.embedding_dim, device=x_t.device)

        hidden_next = self.sample_transition(self.hidden, token_embedding)
        logits = self.decoder(hidden_next)

        self.hidden = hidden_next
        return logits

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, x):
        B, T = x.size()
        device = x.device

        if self.hidden is None:
            h = torch.randn(B, self.embedding_dim, device=device)
        else:
            h = self.hidden

        logits = torch.empty(B, T, self.decoder.out_features, device=device)
        diff_losses = torch.empty(T, device=device)

        for t in range(T):
            token_embedding = self.embedding(x[:, t])

            hidden_next = self.sample_transition(h, token_embedding)

            logits[:, t] = self.decoder(hidden_next)

            diff_losses[t] = self.diffusion_loss(
                hidden_next.detach(),
                h,
                token_embedding
            )

            h = hidden_next

        self.hidden = h
        return logits, diff_losses.mean()
