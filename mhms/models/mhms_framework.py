import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_segmentation import HierarchicalBERT
from models.video_temporal_segmentation import VTS
from models.summarization import ExtractiveSummarizer

class MHMS(nn.Module):
    """
    The Multimodal Hierarchical Multimedia Summarization (MHMS) framework.
    Combines Textual Segmentation, Video Segmentation, and Extractive Summarization
    under a unified model aligned via Optimal Transport.
    """
    def __init__(self, 
                 bert_model_name='bert-base-uncased', 
                 text_hidden_size=768, 
                 visual_feature_dim=1024, 
                 video_hidden_dim=256,
                 video_omega_b=3):
        super(MHMS, self).__init__()
        
        # 1. Modality Segmenters (Feature Extractors)
        self.text_segmenter = HierarchicalBERT(
            pretrained_model_name=bert_model_name, 
            num_article_layers=2, # lightweight for demo
            hidden_size=text_hidden_size
        )
        
        self.video_segmenter = VTS(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=video_hidden_dim,
            omega_b=video_omega_b
        )
        
        # 2. Projectors to a common semantic alignment dimension (Optimal Transport space)
        self.alignment_dim = 256
        self.text_projector = nn.Linear(text_hidden_size, self.alignment_dim)
        
        # Video segmenter VTS outputs LSTM hidden features which we can project directly
        # Wait, VTS forward returns probs. We need to extract the lstm spatial features in VTS.
        # But we can also just project the raw visual features directly or build another sequence encoder.
        # Since we use simple video features, let's encode the sequence linearly.
        self.video_projector = nn.Linear(visual_feature_dim, self.alignment_dim)
        
        # 3. Summarization Modules
        self.text_summarizer = ExtractiveSummarizer(input_dim=text_hidden_size, hidden_dim=text_hidden_size//2)
        
        # For video summarization, we take the raw visual features mapped to hidden
        self.video_summarizer = ExtractiveSummarizer(input_dim=visual_feature_dim, hidden_dim=video_hidden_dim)

    def compute_sinkhorn_loss_torch(self, E, V, reg=0.05, num_iters=10):
        """
        PyTorch differentiable Sinkhorn Optimal Transport.
        Args:
            E: Text embeddings shape (Batch, Seq_Text, Dim)
            V: Video embeddings shape (Batch, Seq_Video, Dim)
            reg: Regularization parameter
            num_iters: number of sinkhorn loops
        Returns:
            ot_loss: Scalar total transport cost
        """
        B, K, D = E.shape
        _, M, _ = V.shape
        
        # Normalize embeddings to compute Cosine Distance
        E_norm = F.normalize(E, p=2, dim=-1)
        V_norm = F.normalize(V, p=2, dim=-1)
        
        # Cosine similarity matrix (B, K, M)
        sim = torch.bmm(E_norm, V_norm.transpose(1, 2))
        
        # Cosine distance cost matrix (B, K, M)
        C = 1.0 - sim
        
        # Uniform marginal distributions (B, K) and (B, M)
        mu = torch.ones(B, K, device=E.device) / K
        nu = torch.ones(B, M, device=E.device) / M
        
        # Gibbs kernel (B, K, M)
        K_matrix = torch.exp(-C / reg)
        
        # Initialize scaling vectors u (B, K) and v (B, M)
        u = torch.ones(B, K, device=E.device) / K
        v = torch.ones(B, M, device=E.device) / M
        
        # Sinkhorn Iterations
        for _ in range(num_iters):
            u = mu / (torch.bmm(K_matrix, v.unsqueeze(2)).squeeze(2) + 1e-8)
            v = nu / (torch.bmm(K_matrix.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)
            
        # Optimal Transport matrix T: diag(u) K_matrix diag(v)
        # Using broadcasting instead of explicit diags
        T = u.unsqueeze(2) * K_matrix * v.unsqueeze(1)
        
        # Compute cost
        distance = torch.sum(T * C) / B # Average over batch
        
        return distance

    def forward(self, input_ids, attention_mask, video_features):
        """
        Args:
            input_ids: (B, Num_Sentences, Max_Words)
            attention_mask: (B, Num_Sentences, Max_Words)
            video_features: (B, Num_Shots, Video_Dim)
            
        Returns:
            dict containing probabilities and OT loss
        """
        
        # 1. Text Segmentation & Feature Extraction
        # Note: Hierarchy BERT might throw error if sequence is too short, handled gracefully
        text_seg_probs, text_features = self.text_segmenter(input_ids, attention_mask)
        
        # 2. Video Segmentation
        video_seg_probs = self.video_segmenter(video_features)
        
        # 3. Summarization
        text_summ_probs = self.text_summarizer(text_features)
        video_summ_probs = self.video_summarizer(video_features)
        
        # 4. Optimal Transport Cross-modal Alignment
        # Project both to Same Alignment Dimension
        E_proj = self.text_projector(text_features)
        V_proj = self.video_projector(video_features)
        
        ot_loss = self.compute_sinkhorn_loss_torch(E_proj, V_proj)
        
        return {
            "text_seg_probs": text_seg_probs,
            "text_features": text_features,
            "video_seg_probs": video_seg_probs,
            "text_summ_probs": text_summ_probs,
            "video_summ_probs": video_summ_probs,
            "ot_loss": ot_loss
        }
