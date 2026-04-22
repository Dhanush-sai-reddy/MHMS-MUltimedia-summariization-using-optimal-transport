import torch
import torch.nn as nn
import torch.nn.functional as F

class VTS(nn.Module):
    def __init__(self, visual_feature_dim=2048, hidden_dim=512, omega_b=5):
        super(VTS, self).__init__()
        self.visual_dim = visual_feature_dim
        self.omega = omega_b
        self.conv_d_before = nn.Conv1d(visual_feature_dim, hidden_dim, kernel_size=omega_b)
        self.conv_d_after = nn.Conv1d(visual_feature_dim, hidden_dim, kernel_size=omega_b)
        self.conv_r = nn.Conv1d(visual_feature_dim, hidden_dim, kernel_size=2 * omega_b)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.bilstm = nn.LSTM(input_size=hidden_dim + 1, hidden_size=hidden_dim,
                              batch_first=True, bidirectional=True)
        self.fc_prob = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, seq_len, D = x.shape
        x_conv = x.permute(0, 2, 1)
        vts_outputs = []

        for center_idx in range(self.omega, seq_len - self.omega):
            window_before = x_conv[:, :, center_idx - self.omega:center_idx]
            window_after = x_conv[:, :, center_idx:center_idx + self.omega]
            full_window = x_conv[:, :, center_idx - self.omega:center_idx + self.omega]

            d_before = self.conv_d_before(window_before).squeeze(-1)
            d_after = self.conv_d_after(window_after).squeeze(-1)
            vts_d = torch.sum(d_before * d_after, dim=-1, keepdim=True)

            r_feat = self.conv_r(full_window)
            vts_r = self.max_pool(r_feat).squeeze(-1)

            vts_outputs.append(torch.cat([vts_r, vts_d], dim=-1))

        if len(vts_outputs) == 0:
            raise ValueError(f"Sequence length {seq_len} too short for omega {self.omega}")

        vts_seq = torch.stack(vts_outputs, dim=1)
        lstm_out, _ = self.bilstm(vts_seq)
        logits = self.fc_prob(lstm_out).squeeze(-1)
        return self.sigmoid(logits)
