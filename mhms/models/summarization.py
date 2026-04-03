import torch
import torch.nn as nn

class ExtractiveSummarizer(nn.Module):
    """
    Extractive summarization head.
    Can be used for both textual sequence encoding and visual sequence encoding.
    It takes a sequence of features (e.g., sentence feature from BERT, or video shot feature)
    and predicts a binary probability of inclusion in the final summary.
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super(ExtractiveSummarizer, self).__init__()
        
        # Bi-LSTM to capture context among sentences or among video shots
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        # Output layer maps the 2*hidden_dim (from Bi-LSTM) to a single sequence score
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence_features):
        """
        Args:
           sequence_features: (Batch Size, Sequence Length, Feature Dim)
        Returns:
           summary_probs: (Batch Size, Sequence Length) probability of each item.
        """
        # Pass features through Bi-LSTM
        # lstm_out shape: (B, Sq_Len, 2 * Hidden)
        lstm_out, _ = self.bilstm(sequence_features)
        
        lstm_out = self.dropout(lstm_out)
        
        # Predict extractive score for each item in the sequence
        # logits shape: (B, Sq_Len, 1) -> squeezed to (B, Sq_Len)
        logits = self.fc(lstm_out).squeeze(-1)
        summary_probs = self.sigmoid(logits)
        
        return summary_probs
