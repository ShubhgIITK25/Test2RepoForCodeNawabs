import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    """
    Attention-based LSTM model for text classification (e.g., spam detection).
    
    The model consists of an embedding layer, a bidirectional LSTM, 
    an attention mechanism to weight the importance of different time steps, 
    and a final linear layer for classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super(AttentionLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM to capture context from both directions
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Since it's bidirectional, the output dimension is hidden_dim * 2
        self.lstm_hidden_dim = hidden_dim * 2
        
        # Attention layer: learns a weight for each time step's hidden state
        self.attention_weights = nn.Linear(self.lstm_hidden_dim, 1)
        
        # Final classification layer
        self.fc = nn.Linear(self.lstm_hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token indices.
            
        Returns:
            tuple: (output, weights)
                - output: Classification logits of shape (batch_size, output_dim).
                - weights: Attention weights of shape (batch_size, seq_len, 1).
        """
        # Embedding layer: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM layer: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, lstm_hidden_dim)
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # Attention mechanism:
        # 1. Compute raw attention scores for each time step: (batch_size, seq_len, 1)
        scores = self.attention_weights(lstm_out)
        
        # 2. Normalize scores using softmax to get weights that sum to 1: (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        
        # 3. Compute the context vector as a weighted sum of LSTM outputs: (batch_size, lstm_hidden_dim)
        context = torch.sum(weights * lstm_out, dim=1)
        
        # 4. Final classification: (batch_size, lstm_hidden_dim) -> (batch_size, output_dim)
        out = self.fc(context)
        
        return out, weights

if __name__ == "__main__":
    # Test the model instantiation and forward pass
    vocab_size = 1000
    embedding_dim = 64
    hidden_dim = 128
    batch_size = 8
    seq_len = 20
    
    model = AttentionLSTM(vocab_size, embedding_dim, hidden_dim)
    
    # Create dummy input data (batch_size, seq_len)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    try:
        output, weights = model(dummy_input)
        print(f"Model successfully instantiated and forward pass completed.")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}") # Expected: (8, 1)
        print(f"Weights shape: {weights.shape}") # Expected: (8, 20, 1)
    except Exception as e:
        print(f"An error occurred: {e}")
