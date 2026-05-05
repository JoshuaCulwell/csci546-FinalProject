import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_2 import DiffusionRNN


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------

DATA_PATH = "./datasets/tiny_shakespeare.txt"

SEQ_LEN = 128
BATCH_SIZE = 512

EMBEDDING_DIM = 512
HIDDEN_DIM = 1024
NUM_DIFF_STEPS = 6

LR = 1e-4
EPOCHS = 10

DIFF_LOSS_WEIGHT = 0.1

SAVE_PATH = "best_model.pt"


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
        return chunk[:-1], chunk[1:]


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Expected dataset at {DATA_PATH}."
        )

    dataset = ShakespeareDataset(DATA_PATH, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DiffusionRNN(
        vocab_size=dataset.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_diff_steps=NUM_DIFF_STEPS,
        device=device,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        for x, y in pbar:
            model.reset_state()
            x, y = x.to(device), y.to(device)

            model.reset_state()

            logits, diff_loss = model(x)

            lm_loss = F.cross_entropy(
                logits.reshape(-1, dataset.vocab_size),
                y.reshape(-1)
            )

            loss = lm_loss + DIFF_LOSS_WEIGHT * diff_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({
                "LM": lm_loss.item(),
                "Diff": diff_loss.item()
            })

        if lm_loss.item() < best_loss:
            best_loss = lm_loss.item()
            torch.save({
                "model_state_dict": model.state_dict(),
                "stoi": dataset.stoi,
                "itos": dataset.itos
            }, SAVE_PATH)
            print(f"Saved new best model with LM loss {best_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    train()
