import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import DiffusionRNN   # <-- your file

# ------------------------------------------------------------
# Toy dataset: random token sequences
# ------------------------------------------------------------

class ToyDataset(Dataset):
    def __init__(self, num_samples=5000, seq_len=16, vocab_size=100):
        self.vocab_size = vocab_size
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_size = 100
    seq_len = 16
    batch_size = 32

    dataset = ToyDataset(num_samples=2000, seq_len=seq_len, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DiffusionRNN(vocab_size=vocab_size, dim=128, num_steps=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        for i, x in enumerate(loader):
            x = x.to(device)

            logits, diff_loss = model(x)

            # LM loss: predict x[t+1] from h[t+1]
            lm_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                x[:, 1:].reshape(-1)
            )

            loss = lm_loss + 0.1 * diff_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {i} | "
                    f"LM Loss: {lm_loss.item():.4f} | "
                    f"Diff Loss: {diff_loss.item():.4f}"
                )


if __name__ == "__main__":
    train()

