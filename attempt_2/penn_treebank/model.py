import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t, dim):
    #Sinusoidal
    device = t.device
    half = dim // 2

    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=device) / half * 10
    )

    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        depth,
        activation_function=nn.ReLU
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        layers = []
        layers.append(nn.LayerNorm(in_dim))
        for i in range(depth):
            dim = hidden_dim if i < depth - 1 else out_dim

            layers.append(nn.Linear(in_dim, dim))

            if i < depth - 1:
                layers.append(activation_function())

            in_dim = dim

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLPTransition(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        width,
        depth,
        activation_function=nn.ReLU,
    ):
        super().__init__()

        self.network = MLP(
            in_dim = in_dim,
            out_dim = hidden_dim,
            hidden_dim = width,
            depth = depth,
            activation_function = activation_function
        )

    def forward(self, hidden_state, x):
        out = torch.cat([hidden_state, x], dim=-1)
        out = self.network(out)
        out = torch.tanh(out)

        return out

class DiffusionTransition(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_diffusion_steps,
        denoising_hidden_dim,
        denoising_depth,
        denoising_activation_function,
        timestep_dim,
        beta_start,
        beta_end,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps

        if timestep_dim % 2 != 0:
            timestep_dim += 1

        self.timestep_dim = timestep_dim

        beta_schedule = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        self.register_buffer("beta_schedule", beta_schedule)
        self.register_buffer("alpha", 1.0 - beta_schedule)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - beta_schedule, dim=0))

        denoiser_in_dim = hidden_dim + in_dim + timestep_dim
        self.denoiser = MLP(
            in_dim = denoiser_in_dim,
            out_dim = hidden_dim,
            hidden_dim = denoising_hidden_dim,
            depth = denoising_depth,
            activation_function = denoising_activation_function
        )

    def forward(self, hidden_state, x):
        batch_size = hidden_state.shape[0]
        device = hidden_state.device

        #noise = torch.randn_like(hidden_state, device=device)
        #new_hidden_state = hidden_state * self.alpha_bar[-1].sqrt()
        #new_hidden_state += noise * (1 - self.alpha_bar[-1]).sqrt()

        new_hidden_state = torch.randn(batch_size, self.hidden_dim, device=device)

        for t in reversed(range(self.num_diffusion_steps)):
            beta_t = self.beta_schedule[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]

            t_vec = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_embedding = timestep_embedding(t_vec, self.timestep_dim)

            denoiser_in = torch.cat([
                new_hidden_state, hidden_state, x, t_embedding
            ], dim=-1)

            noise_prediction = self.denoiser(denoiser_in)

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            mean = coef1 * (new_hidden_state - coef2 * noise_prediction)

            if t > 0:
                noise = torch.randn_like(new_hidden_state)
                new_hidden_state = mean + torch.sqrt(beta_t) * noise
            else:
                new_hidden_state = mean

        return torch.tanh(new_hidden_state)

class VariableTransitionRNN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_state_dim,
        embedding_dim,
        transition_module,
        out_mlp_width,
        out_mlp_depth,
        out_mlp_activation_function,
        vocab_size
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_state_dim = hidden_state_dim
        self.transition = transition_module
        self.out_mlp = MLP(
            hidden_state_dim,
            out_dim,
            out_mlp_width,
            out_mlp_depth,
            out_mlp_activation_function,
        )

    def forward(self, input_sequence, initial_hidden_state=None):
        batch_size, sequence_length = input_sequence.shape

        input_sequence = self.embedding(input_sequence)

        if initial_hidden_state is None:
            hidden_state = torch.zeros(
                batch_size,
                self.hidden_state_dim,
                device=input_sequence.device
            )
        else:
            hidden_state = initial_hidden_state

        outputs = []
        for i in range(sequence_length):
            current_input = input_sequence[:, i, :]

            hidden_state = self.transition(
                hidden_state = hidden_state,
                x = current_input,
            )

            output = self.out_mlp(hidden_state)
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)











if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 4
    seq_len = 5
    input_dim = 1
    embedding_dim = 4
    timestep_dim = 8
    hidden_dim = 16
    out_dim = 10

    # Dummy input sequence
    x = torch.randn(batch_size, seq_len, input_dim)

    print("\n=== Testing MLPTransition ===")
    mlp_transition = MLPTransition(
        in_dim=input_dim,
        hidden_dim=hidden_dim,
        input_embedding_dim=embedding_dim,
        width=32,
        depth=2,
        activation_function=nn.ReLU,
    )

    rnn_mlp = VariableTransitionRNN(
        in_dim=input_dim,
        out_dim=out_dim,
        hidden_state_dim=hidden_dim,
        transition_module=mlp_transition,
        out_mlp_width=32,
        out_mlp_depth=2,
        out_mlp_activation_function=nn.ReLU,
    )

    out_mlp = rnn_mlp(x)
    print("Output shape (MLPTransition):", out_mlp.shape)


    print("\n=== Testing DiffusionTransition ===")
    diffusion_transition = DiffusionTransition(
        in_dim=input_dim,
        hidden_dim=hidden_dim,
        input_embedding_dim=embedding_dim,
        num_diffusion_steps=4,
        denoising_hidden_dim=32,
        denoising_depth=2,
        denoising_activation_function=nn.ReLU,
        timestep_dim=timestep_dim,
        beta_start=1e-4,
        beta_end=0.02,
    )

    rnn_diff = VariableTransitionRNN(
        in_dim=input_dim,
        out_dim=out_dim,
        hidden_state_dim=hidden_dim,
        transition_module=diffusion_transition,
        out_mlp_width=32,
        out_mlp_depth=2,
        out_mlp_activation_function=nn.ReLU,
    )

    out_diff = rnn_diff(x)
    print("Output shape (DiffusionTransition):", out_diff.shape)

