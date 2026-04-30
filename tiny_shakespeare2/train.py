import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import DiffusionRNN


# ============================================================
# Hyperparameters (EDIT HERE)
# ============================================================

DATA_PATH = "./datasets/tiny_shakespeare/input.txt"

SEQ_LEN = 128
BATCH_SIZE = 512

DIM = 1024
NUM_DIFFUSION_STEPS = 6

LR = 1e-4
EPOCHS = 10

DIFF_LOSS_WEIGHT = 0.1

USE_COMPILE = False  # torch.compile
NUM_WORKERS = 4

SAVE_PATH = "best_model.pt"


# ============================================================
# Dataset
# ============================================================

class ShakespeareDataset(Dataset):
    def __init__(self, path, seq_len):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(set(text))
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


# ============================================================
# Training
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    dataset = ShakespeareDataset(DATA_PATH, SEQ_LEN)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = DiffusionRNN(
        vocab_size=dataset.vocab_size,
        dim=DIM,
        num_steps=NUM_DIFFUSION_STEPS
    ).to(device)

    if USE_COMPILE:
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()

        total_lm = 0.0
        total_diff = 0.0
        steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits, diff_loss = model(x)

            lm_loss = F.cross_entropy(
                logits.view(-1, dataset.vocab_size),
                y.view(-1)
            )

            loss = lm_loss + DIFF_LOSS_WEIGHT * diff_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # tracking
            total_lm += lm_loss.item()
            total_diff += diff_loss.item()
            steps += 1

            pbar.set_postfix({
                "LM": total_lm / steps,
                "Diff": total_diff / steps
            })

        avg_lm = total_lm / steps

        # Save best model
        if avg_lm < best_loss:
            best_loss = avg_lm
            torch.save({
                "model_state_dict": model.state_dict(),
                "stoi": dataset.stoi,
                "itos": dataset.itos,
            }, SAVE_PATH)

            print(f"✅ Saved new best model (LM loss: {best_loss:.4f})")

    print("Training complete.")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    train()
