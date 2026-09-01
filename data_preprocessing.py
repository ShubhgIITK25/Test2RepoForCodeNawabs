import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter, defaultdict
import re
import os
import urllib.request
import zipfile
import random

def tokenize(text):
    """Basic tokenization: lowercase and split by non-alphanumeric characters."""
    return re.findall(r'\w+', text.lower())

def build_vocab(texts, max_vocab_size=10000):
    """Builds a vocabulary mapping from words to indices."""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    
    # Most common words
    most_common = counter.most_common(max_vocab_size - 2) # -2 for <PAD> and <UNK>
    vocab = {word: i + 2 for i, (word, _) in enumerate(most_common)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def text_to_sequence(text, vocab, max_len):
    """Converts a text string to a padded sequence of vocabulary indices."""
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Padding or truncating
    if len(sequence) < max_len:
        sequence += [vocab['<PAD>']] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    
    return sequence

def simple_train_test_split(texts, labels, test_size=0.2, random_state=42):
    """Simple train-test split without sklearn."""
    random.seed(random_state)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    
    test_size = int(len(indices) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_texts = [texts[i] for i in train_indices]
    test_texts = [texts[i] for i in test_indices]
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_texts, test_texts, train_labels, test_labels

def download_dataset(dest_path='SMSSpamCollection'):
    """Downloads the SMS Spam Collection dataset if it doesn't exist."""
    if os.path.exists(dest_path):
        return dest_path
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "smsspamcollection.zip"
    
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    os.remove(zip_path)
    return 'SMSSpamCollection'

def load_and_preprocess_data(filepath, max_len=100, max_vocab_size=10000, test_size=0.2, random_state=42):
    """Loads, tokenizes, pads, and splits the SMS spam dataset."""
    # Load dataset using standard library
    # The SMS Spam Collection is a TSV file without headers: label \t message
    texts = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                label = parts[0]
                text = '\t'.join(parts[1:])  # Handle cases where message contains tabs
                labels.append(1 if label == 'spam' else 0)  # spam=1, ham=0
                texts.append(text)
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = simple_train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )
    
    # Build vocab from training set only to avoid leakage
    vocab = build_vocab(train_texts, max_vocab_size=max_vocab_size)
    
    # Process texts to sequences
    X_train = [text_to_sequence(t, vocab, max_len) for t in train_texts]
    X_test = [text_to_sequence(t, vocab, max_len) for t in test_texts]
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.long)
    y_test = torch.tensor(test_labels, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test, vocab

def get_dataloaders(filepath=None, batch_size=32, max_len=100, max_vocab_size=10000):
    """Returns PyTorch DataLoaders for training and testing."""
    if filepath is None:
        filepath = download_dataset()
        
    X_train, y_train, X_test, y_test, vocab = load_and_preprocess_data(
        filepath, max_len=max_len, max_vocab_size=max_vocab_size
    )
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, vocab

if __name__ == "__main__":
    # Test the preprocessing pipeline
    try:
        train_loader, test_loader, vocab = get_dataloaders()
        print(f"Vocab size: {len(vocab)}")
        
        # Check a batch
        X_batch, y_batch = next(iter(train_loader))
        print(f"Batch X shape: {X_batch.shape}") # [batch_size, max_len]
        print(f"Batch y shape: {y_batch.shape}") # [batch_size]
        print("Preprocessing successful!")
    except Exception as e:
        print(f"Preprocessing failed: {e}")