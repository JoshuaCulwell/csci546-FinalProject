import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import DiffusionRNN


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class ShakespeareDataset(Dataset):
    def __init__(self, path, seq_len=128):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # input, target


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = "./datasets/tiny_shakespeare/input.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Expected dataset at {data_path}. "
            "Download tiny shakespeare and place it there."
        )

    dataset = ShakespeareDataset(data_path, seq_len=128)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = DiffusionRNN(
        vocab_size=dataset.vocab_size,
        dim=1024,
        num_steps=6
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float("inf")
    save_path = "best_model.pt"

    for epoch in range(10):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            logits, diff_loss = model(x)

            lm_loss = F.cross_entropy(
                logits.reshape(-1, dataset.vocab_size),
                y.reshape(-1)
            )

            loss = lm_loss + 0.1 * diff_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({
                "LM": lm_loss.item(),
                "Diff": diff_loss.item()
            })

        # Save best model
        if lm_loss.item() < best_loss:
            best_loss = lm_loss.item()
            torch.save({
                "model_state_dict": model.state_dict(),
                "stoi": dataset.stoi,
                "itos": dataset.itos
            }, save_path)
            print(f"Saved new best model with LM loss {best_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    train()

