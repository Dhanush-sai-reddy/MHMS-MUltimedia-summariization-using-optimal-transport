import torch
import torch.nn as nn
import torch.nn.functional as F

class TextExtractiveSummarizer(nn.Module):
    """
    Lightweight extractive summarization head for text.
    Relies on predicting sequence significance directly from HierarchicalBERT encodings.
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super(TextExtractiveSummarizer, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence_features):
        lstm_out, _ = self.bilstm(sequence_features)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out).squeeze(-1)
        return self.sigmoid(logits)


class VisualEncoderDecoderSummarizer(nn.Module):
    """
    Visual Summarization module based on Section 3.2 of the MHMS paper.
    It uses an Encoder (Bi-LSTM) and Decoder (LSTM) architecture with an 
    attention mechanism to capture temporal ordering and dependency 
    to generate importance scores for each frame/shot.
    """
    def __init__(self, input_dim, hidden_dim=256):
        super(VisualEncoderDecoderSummarizer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder: Bi-LSTM
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                               batch_first=True, bidirectional=True)
        
        # Decoder: LSTM 
        # Decoder input is the context vector E_t (dim = hidden_dim * 2) plus previous score d_{t-1} (dim = 1)
        self.decoder = nn.LSTMCell(input_size=(hidden_dim * 2) + 1, hidden_size=hidden_dim)
        
        # Attention mechanism parameters (Eq 7)
        self.W_a = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        
        # Final projection to score
        self.fc_score = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, video_features):
        """
        video_features: (B, M, input_dim) where M is sequence length (num shots)
        Returns:
            scores: (B, M)
        """
        B, M, _ = video_features.shape
        device = video_features.device
        
        # 1. Encode
        # E shape: (B, M, 2 * hidden_dim)
        E, (h_n, c_n) = self.encoder(video_features)
        
        # 2. Decode auto-regressively to generate importance sequence
        scores = torch.zeros(B, M, device=device)
        
        # Initialize decoder state with the final forward/backward states of encoder
        # h_n is (2, B, hidden_dim). We can just take the backward or forward, or sum them.
        hx = h_n[0] + h_n[1] # (B, hidden_dim)
        cx = c_n[0] + c_n[1] # (B, hidden_dim)
        
        # In this extractive formulation, we need M output scores.
        for t in range(M):
            # Attention Mechanism (Eq 6, 7, 8)
            # score function: e_t^i = e_i^T W_a s_{t-1}
            # hx is conceptually our s_{t-1} state of the decoder: (B, hidden_dim)
            
            # W_a(hx) -> (B, 2 * hidden_dim)
            query = self.W_a(hx).unsqueeze(2) # (B, 2*hidden_dim, 1)
            
            # E is (B, M, 2*hidden_dim)
            # bmm(E, query) -> (B, M, 1)
            attn_energies = torch.bmm(E, query).squeeze(2) # (B, M)
            alpha_t = F.softmax(attn_energies, dim=1)      # (B, M)
            
            # Context vector E_t: (B, 2*hidden_dim)
            # bmm(alpha_t.unsqueeze(1), E) -> (B, 1, M) x (B, M, 2*hidden_dim) -> (B, 1, 2*hidden_dim)
            context_t = torch.bmm(alpha_t.unsqueeze(1), E).squeeze(1)
            
            # Previous score d_{t-1}
            if t == 0:
                d_prev = torch.zeros(B, 1, device=device)
            else:
                d_prev = scores[:, t-1].unsqueeze(1)
                
            # Decoder step: inputs [context_t, d_{t-1}]
            decoder_input = torch.cat([context_t, d_prev], dim=1) # (B, 2*hidden_dim + 1)
            hx, cx = self.decoder(decoder_input, (hx, cx))
            
            # Project to single probability score d_t
            score_t = self.sigmoid(self.fc_score(hx).squeeze(-1)) # (B)
            scores[:, t] = score_t
            
        return scores
