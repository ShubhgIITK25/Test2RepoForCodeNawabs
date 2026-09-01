import argparse
import os

import torch
import torch.nn as nn

from data_preprocessing import get_dataloaders
from spam_model import AttentionLSTM


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch. Returns (average loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        output, _ = model(X_batch)
        loss = criterion(output.squeeze(1), y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        predictions = (torch.sigmoid(output.squeeze(1)) >= 0.5).float()
        correct += (predictions == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model on a dataset. Returns (average loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        output, _ = model(X_batch)
        loss = criterion(output.squeeze(1), y_batch)

        total_loss += loss.item() * X_batch.size(0)
        predictions = (torch.sigmoid(output.squeeze(1)) >= 0.5).float()
        correct += (predictions == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the attention-based LSTM spam classifier.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--max-len", type=int, default=100, help="Max sequence length")
    parser.add_argument("--max-vocab-size", type=int, default=10000, help="Max vocabulary size")
    parser.add_argument("--data", type=str, default=None, help="Path to SMSSpamCollection file")
    parser.add_argument("--save-path", type=str, default="spam_model.h5", help="Where to save model weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader, vocab = get_dataloaders(
        filepath=args.data,
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_vocab_size=args.max_vocab_size,
    )
    print(f"Vocab size: {len(vocab)}")

    # Build model
    model = AttentionLSTM(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1,
    ).to(device)
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
            f"test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f}"
        )
        if test_acc >= best_test_acc:
            best_test_acc = test_acc

    # Save model weights
    torch.save(model.state_dict(), args.save_path)
    print(f"Model weights saved to {os.path.abspath(args.save_path)}")
    print(f"Best test accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()