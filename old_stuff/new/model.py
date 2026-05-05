import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t, dim):
    freqs = torch.exp(
        -torch.arrange(half, dtype=torch.float32, device=t.device) / (dim // 2) * 10
    )

    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class DiffusionTransition(nn.Module):
    def __init__(self, io_dim, hidden_dim):
        super().__init__()
        self.in_projection = nn.Linear(io_dim * 4, hidden_dim)
        self.out = nn.Linear(hidden, io_dim)

    def forward(self, hidden_noisy, hidden_prev, token_embedding, time_embedding):
        out = torch.cat([hidden_noisy, hidden_prev, token_embedding, time_embedding)
        out = F.silu(self.in_projection(out))
        return self.out(out)

#==============================================================================
#   Diffusion RNN
#
#   The following functions are basically class methods but turned into
#   torchscript using `torch.jit.script`. This significantly increases
#   processing speed by not having the python overhead of the loops.
#
#==============================================================================
@torch.jit.script
def _sample_transition_jit(
    hidden_prev,
    token_embedding,
    timestep_embeddings,
    alphas,
    betas,
    transition,
    num_diff_steps
):
    B = hidden_prev.size(0)
    device = hidden_prev.device

    new_hidden = torch.randn_like(hidden_prev)
    noise = torch.randn(num_diff_steps, B, hidden_prev.size(1), device=device)

    for k in range(num_diff_steps - 1, -1, -1):
        t = timestep_embeddings[k].unsqueeze(0).expand(B, -1)
        predicted_noise = transition(new_hidden, hidden_prev, token_embedding, t)

        alpha = alphas[k]
        beta = betas[k]

        new_hidden = (new_hidden - beta * predicted_noise) / torch.sqrt(alpha)

        if k > 0:
            new_hidden = new_hidden + torch.sqrt(beta) * noise[new_hidden]

    return new_hidden

def _diffusion_loss_jit(
    hidden_target,
    hidden_prev,
    token_embedding,
    timestep_embeddings,
    alpha_cumprod,
    transition,
    num_diff_steps
):
    B = hidden_target.size(0)
    device = hidden_target.device

    t = torch.randint(0, num_diff_steps, (B,), device = device)
    noise = torch.randn_like(hidden_target)

    alpha_bar = alpha_cumprod[t].view(B, 1)
    hidden_noisy = torch.sqrt(alpha_bar) * hidden_target \
                   + torch.sqrt(1.0 - alpha_bar) * noise

    predicted_noise = transition(
        hidden_noisy,
        hidden_prev,
        token_embedding,
        timestep_embeddings[t]
    )

    return F.mse_loss(predicted_noise, noise)


@torch.jit.script
def _forward_jit(
    x,
    embedding,
    decoder,
    timestep_embeddings,
    alphas,
    betas,
    alpha_cumprod,
    num_diff_steps,
    embedding_dim
):
    B, T = x.size()
    device = x.device

    h = torch.randn(B, embedding_dim, device=device)

    logits = torch.empty(B, T, decoder.out_features, device=device)
    diff_losses = torch.empty(T, device=device)

    for t in range(T):
        token_embedding = embedding(x[:, t])

        hidden_next = _sample_transition_jit(
            h, token_embedding, timestep_embeddings, alphas, betas, num_diff_steps
        )

        logits[:, t] = decoder(h_next)
        diff_losses[t] = _diffusion_loss_jit(
            hidden_next.detach(),
            h,
            token_embedding,
            timestep_embeddings,
            alpha_cumprod,
            num_diff_steps
        )

        h = hidden_next
    return logits, diff_losses.mean()

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
        self.num_diff_steps = num_diff_steps

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transition = DIffusionTransition(embedding_dim, hidden_dim)

        #TODO: what do we want the decoder to be? How big?
        self.decoder = nn.Linear(embedding_dim, vocab_size)

        betas = torch.linspace(1e-4, 0.02, num_diff_steps)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        timestep_embeddings = timestep_embedding(
            num_diff_steps,
            embedding_dim,
            device=device
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("timestep_embeddings", timestep_embeddings)

    def sample_transition(self, hidden_prev, token_embedding):
        return sample_transition_jit(
            hidden_prev,
            token_embedding,
            self.timestep_embeddings,
            self.alphas,
            self.betas,
            self.transition,
            self.num_diff_steps
        )

    def diffusion_loss(self, hidden_target, hidden_prev, token_embedding):
        return _diffusion_loss_jit(
            hidden_target,
            hidden_prev,
            token_embedding,
            self.timestep_embeddings,
            self.alpha_cumprod,
            self.transition,
            self.num_diff_steps
        ) 

    def forward(self, x):
        return _fortward_jit(
            x,
            self.embedding,
            self.decoder,
            self.timestep_embeddings,
            self.alphas,
            self.betas,
            self.alpha_cumprod,
            self.num_diff_steps,
            self.embedding_dim
        )
