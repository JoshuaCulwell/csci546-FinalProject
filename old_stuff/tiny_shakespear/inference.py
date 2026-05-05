import torch
from model import DiffusionRNN


def load_model(path="best_model.pt"):
    checkpoint = torch.load(path, map_location="cpu")

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = len(stoi)

    model = DiffusionRNN(vocab_size=vocab_size, dim=1024, num_steps=6)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, stoi, itos


def generate(model, stoi, itos, prompt, steps=200):
    device = next(model.parameters()).device

    # Encode prompt
    x = torch.tensor([stoi.get(c, 0) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)

    h = torch.randn(1, model.dim).to(device)

    # Warm up hidden state with prompt
    for i in range(x.size(1)):
        tok_embed = model.embed(x[:, i])
        h = model.sample_transition(h, tok_embed)

    out = list(prompt)

    # Generate new characters
    for _ in range(steps):
        logits = model.decoder(h)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = torch.multinomial(probs, num_samples=1).item()

        out.append(itos[idx])

        tok = torch.tensor([idx], device=device)
        tok_embed = model.embed(tok)
        h = model.sample_transition(h, tok_embed)

    return "".join(out)


def main():
    model, stoi, itos = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    while True:
        prompt = input("\nEnter prompt: ")
        print("\nGenerating...\n")
        print(generate(model, stoi, itos, prompt))


if __name__ == "__main__":
    main()

