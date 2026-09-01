import torch
import argparse
import os
from data_preprocessing import get_dataloaders
from spam_model import AttentionLSTM

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1-score.
    y_true and y_pred are torch tensors of shape (n,) containing 0 or 1.
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate the attention-based LSTM spam classifier.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--max-len", type=int, default=100, help="Max sequence length")
    parser.add_argument("--max-vocab-size", type=int, default=10000, help="Max vocabulary size")
    parser.add_argument("--data", type=str, default=None, help="Path to SMSSpamCollection file")
    parser.add_argument("--model-path", type=str, default="spam_model.h5", help="Path to saved model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    # We only need the test_loader and vocab for evaluation
    _, test_loader, vocab = get_dataloaders(
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

    # Load model weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model weights loaded from {args.model_path}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            output, _ = model(X_batch)
            predictions = (torch.sigmoid(output.squeeze(1)) >= 0.5).float()
            
            all_preds.append(predictions.cpu())
            all_labels.append(y_batch.cpu())

    # Concatenate all batches
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_labels)

    # Compute metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)

    print("\nEvaluation Results:")
    print("-" * 20)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-" * 20)

if __name__ == "__main__":
    main()
