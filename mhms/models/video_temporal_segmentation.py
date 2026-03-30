import torch
import torch.nn as nn
import torch.nn.functional as F

class VTS(nn.Module):
    """
    Video Temporal Segmentation Module as described in MHMS paper (Section 3.1).
    It splits the original video into small segments (scenes/shots).
    
    The architecture relies on VTS_d (Difference) and VTS_r (Relation) paths, followed
    by a Bi-LSTM that predicts a sequence of coarse probability scores s_i to dictate boundaries.
    """
    def __init__(self, visual_feature_dim=2048, hidden_dim=512, omega_b=5):
        super(VTS, self).__init__()
        
        self.visual_dim = visual_feature_dim
        self.omega = omega_b
        
        # Branch VTS_d: Difference
        # Two temporal convolutions that embed the omega_b shots before and after boundary
        self.conv_d_before = nn.Conv1d(in_channels=visual_feature_dim, out_channels=hidden_dim, kernel_size=omega_b)
        self.conv_d_after  = nn.Conv1d(in_channels=visual_feature_dim, out_channels=hidden_dim, kernel_size=omega_b)
        
        # Branch VTS_r: Relation 
        # Temporal convolution followed by max pooling over the whole window (2 * omega_b)
        self.conv_r = nn.Conv1d(in_channels=visual_feature_dim, out_channels=hidden_dim, kernel_size=2 * omega_b)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Bi-LSTM to predict sequence of coarse scores
        # Input features to Bi-LSTM: VTS_d (scalar diff) + VTS_r (hidden_dim)
        # We concatenate VTS_r and VTS_d into one vector per boundary
        self.bilstm = nn.LSTM(
            input_size=hidden_dim + 1,  # 1 for inner product difference scalar
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer to project Bi-LSTM sequence into boundary probabilities
        self.fc_prob = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Calculates boundary probabilities for a given batch of video shot sequences.
        Args:
           x: Video features of shape (Batch Size, Sequence Length, Feature Dim). 
              For instance, ResNet-101 features.
        Returns:
           probs: Boundary probabilities (Batch Size, Sequence Length - 2*omega)
        """
        B, seq_len, D = x.shape
        
        # Convert to Conv1D layout: (B, Channels, Length)
        x_conv = x.permute(0, 2, 1)
        
        vts_outputs = []
        
        # We slide a window of size 2 * omega_b to find boundaries
        # In a real batched implementation, this loop can be vectorized using nn.Unfold or grouped Convs.
        # This loop mimics the behavior of scoring each boundary candidate chronologically.
        for center_idx in range(self.omega, seq_len - self.omega):
            
            # Select the window: omega_b frames before and omega_b frames after the boundary
            window_before = x_conv[:, :, center_idx - self.omega : center_idx]
            window_after  = x_conv[:, :, center_idx : center_idx + self.omega]
            full_window   = x_conv[:, :, center_idx - self.omega : center_idx + self.omega]
            
            # VTS_d Branch
            d_before = self.conv_d_before(window_before)  # (B, Hidden, 1)
            d_after  = self.conv_d_after(window_after)    # (B, Hidden, 1)
            
            # Inner product calculation for differences
            d_before_squeeze = d_before.squeeze(-1)
            d_after_squeeze  = d_after.squeeze(-1)
            
            # Dot product across hidden dimension: (B, 1)
            vts_d = torch.sum(d_before_squeeze * d_after_squeeze, dim=-1, keepdim=True)
            
            # VTS_r Branch
            r_feat = self.conv_r(full_window)  # (B, Hidden, 1)
            vts_r = self.max_pool(r_feat).squeeze(-1)  # (B, Hidden)
            
            # Concatenate inner product and relation representation
            joint_repr = torch.cat([vts_r, vts_d], dim=-1) # (B, Hidden + 1)
            vts_outputs.append(joint_repr)
            
        # Stack the sequence back together: (B, Valid Sequence Length, Hidden + 1)
        if len(vts_outputs) == 0:
            raise ValueError(f"Sequence length {seq_len} is too short for omega {self.omega}")
            
        vts_seq = torch.stack(vts_outputs, dim=1)
        
        # Pass through Bi-LSTM Sequence Modeler
        lstm_out, _ = self.bilstm(vts_seq)  # (B, Valid Sequence Length, 2 * Hidden)
        
        # Coarse score prediction (s_i)
        logits = self.fc_prob(lstm_out).squeeze(-1)  # (B, Valid Sequence Length)
        probs = self.sigmoid(logits)
        
        # For prediction \hat{P}_{vi}, you threshold the probs tensor later with \tau = 0.5.
        
        return probs
