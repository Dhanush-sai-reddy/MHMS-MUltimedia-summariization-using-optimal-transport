import torch
import torch.nn as nn
import torch.nn.functional as F

class TextExtractiveSummarizer(nn.Module):
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
        return self.sigmoid(self.fc(lstm_out).squeeze(-1))


class VisualEncoderDecoderSummarizer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(VisualEncoderDecoderSummarizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(input_size=(hidden_dim * 2) + 1, hidden_size=hidden_dim)
        self.W_a = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.fc_score = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, video_features):
        B, M, _ = video_features.shape
        device = video_features.device

        E, (h_n, c_n) = self.encoder(video_features)
        scores = torch.zeros(B, M, device=device)
        hx = h_n[0] + h_n[1]
        cx = c_n[0] + c_n[1]

        for t in range(M):
            query = self.W_a(hx).unsqueeze(2)
            attn_energies = torch.bmm(E, query).squeeze(2)
            alpha_t = F.softmax(attn_energies, dim=1)
            context_t = torch.bmm(alpha_t.unsqueeze(1), E).squeeze(1)

            d_prev = torch.zeros(B, 1, device=device) if t == 0 else scores[:, t-1].unsqueeze(1)
            decoder_input = torch.cat([context_t, d_prev], dim=1)
            hx, cx = self.decoder(decoder_input, (hx, cx))
            scores[:, t] = self.sigmoid(self.fc_score(hx).squeeze(-1))

        return scores
