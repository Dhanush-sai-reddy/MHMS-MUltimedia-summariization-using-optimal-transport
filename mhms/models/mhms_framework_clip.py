"""
MHMS Framework with Qwen3 VL Unified Embeddings
================================================
Modified MHMS framework optimized for Qwen3 VL embeddings where:
- Text and visual features share the SAME embedding dimension
- Cross-modal alignment via Optimal Transport is more effective
- Unified semantic space enables better multimodal understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mhms.models.video_temporal_segmentation import VTS
from mhms.models.text_segmentation import HierarchicalBERT
from mhms.models.summarization import TextExtractiveSummarizer, VisualEncoderDecoderSummarizer


class MHMS_CLIP(nn.Module):
    """
    Multimodal Hierarchical Multimedia Summarization with Qwen3 VL embeddings.
    
    Key improvements:
    - Unified embedding space (both modalities: same dimension)
    - More effective Optimal Transport alignment
    - Better cross-modal semantic understanding
    """
    
    def __init__(self,
                 embedding_dim=2560,  # Qwen3-VL-4B hidden size (4096 for 8B-Instruct)
                 video_hidden_dim=512,
                 text_hidden_dim=512,
                 video_omega_b=3,
                 use_text_segmentation=False):
        super(MHMS_CLIP, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 1. Text Segmentation (HierarchicalBERT) - Optional, disabled by default for speed
        self.use_text_segmentation = use_text_segmentation
        if use_text_segmentation:
            self.text_segmenter = HierarchicalBERT(
                pretrained_model_name='bert-base-uncased',
                num_article_layers=12,
                hidden_size=embedding_dim,  # Match Qwen dim
                num_heads=12
            )
        
        # 2. Video Segmentation (VTS) - Per paper Section 3.1
        # Note: VTS expects visual_feature_dim, but we use unified Qwen dim
        self.video_segmenter = VTS(
            visual_feature_dim=embedding_dim,
            hidden_dim=video_hidden_dim,
            omega_b=video_omega_b
        )
        
        # 3. Projectors to alignment dimension
        # Since both modalities share the same input dim, projectors can be simpler
        self.alignment_dim = 256
        self.text_projector = nn.Linear(embedding_dim, self.alignment_dim)
        self.video_projector = nn.Linear(embedding_dim, self.alignment_dim)
        
        # 4. Summarization Modules
        self.text_summarizer = TextExtractiveSummarizer(
            input_dim=embedding_dim, 
            hidden_dim=text_hidden_dim
        )
        self.video_summarizer = VisualEncoderDecoderSummarizer(
            input_dim=embedding_dim,  # Unified dim!
            hidden_dim=video_hidden_dim
        )

    def compute_sinkhorn_loss_torch(self, E, V, reg=0.05, num_iters=10):
        """
        PyTorch differentiable Sinkhorn Optimal Transport.
        Now with unified embeddings, the alignment is more semantically meaningful.
        """
        B, K, D = E.shape
        _, M, _ = V.shape
        
        # Normalize embeddings to compute Cosine Distance
        E_norm = F.normalize(E, p=2, dim=-1, eps=1e-8)
        V_norm = F.normalize(V, p=2, dim=-1, eps=1e-8)
        
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
            
        # Optimal Transport matrix T
        T = u.unsqueeze(2) * K_matrix * v.unsqueeze(1)
        
        # Compute cost
        distance = torch.sum(T * C) / B  # Average over batch
        
        return distance, T

    def forward(self, text_features, video_features, text_input_ids=None, text_attention_mask=None):
        """
        Forward pass through the MHMS framework with Qwen3 VL embeddings.
        
        Args:
            text_features: (B, Num_Sentences, embedding_dim) - Qwen3 VL unified embeddings
            video_features: (B, Num_Shots, embedding_dim) - Qwen3 VL unified embeddings
            
        Returns:
            dict containing all probabilities and losses
        """
        B, S, _ = text_features.shape
        
        # 1. Text Segmentation (optional - HierarchicalBERT)
        if self.use_text_segmentation and text_input_ids is not None:
            text_seg_probs, _ = self.text_segmenter(text_input_ids, text_attention_mask)
        else:
            # Dummy segmentation (equal probability)
            text_seg_probs = torch.ones(B, S, device=text_features.device) * 0.5
        
        # 2. Video Segmentation (VTS)
        video_seg_probs = self.video_segmenter(video_features)
        
        # 3. Summarization
        text_summ_probs = self.text_summarizer(text_features)
        video_summ_probs = self.video_summarizer(video_features)
        
        # 4. Cross-Domain Alignment via Optimal Transport
        # With unified embeddings, this alignment is more semantically meaningful!
        E_weighted = text_features * text_summ_probs.unsqueeze(-1)
        V_weighted = video_features * video_summ_probs.unsqueeze(-1)
        
        # Project to common alignment space
        E_proj = self.text_projector(E_weighted)
        V_proj = self.video_projector(V_weighted)
        
        # Compute Sinkhorn OT loss
        ot_loss, T_matrix = self.compute_sinkhorn_loss_torch(E_proj, V_proj)
        
        return {
            "text_seg_probs": text_seg_probs,
            "text_features": text_features,
            "video_seg_probs": video_seg_probs,
            "text_summ_probs": text_summ_probs,
            "video_summ_probs": video_summ_probs,
            "ot_loss": ot_loss,
            "ot_alignment_matrix": T_matrix
        }

    def generate_multimodal_summary(self, text_features, video_features, threshold=0.5):
        """
        Generate multimodal summary using Optimal Transport alignment.
        With unified embeddings, the alignment is more accurate.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(text_features, video_features)
            
            text_probs = outputs["text_summ_probs"]
            video_probs = outputs["video_summ_probs"]
            T_matrix = outputs["ot_alignment_matrix"]
            
            matched_summaries = []
            B = T_matrix.shape[0]
            
            for b in range(B):
                # Select candidates via threshold
                text_candidates = torch.where(text_probs[b] > threshold)[0]
                video_candidates = torch.where(video_probs[b] > threshold)[0]
                
                # Align using Transport Plan T
                alignments = []
                for t_idx in text_candidates:
                    for v_idx in video_candidates:
                        match_score = T_matrix[b, t_idx, v_idx].item()
                        alignments.append({
                            "text_idx": t_idx.item(),
                            "video_idx": v_idx.item(),
                            "match_score": match_score
                        })
                        
                alignments.sort(key=lambda x: x["match_score"], reverse=True)
                matched_summaries.append(alignments)
                
            return matched_summaries

    def generate_multimodal_summary_topk(self, text_features, video_features, top_k=3):
        """
        Adaptive Top-K inference with unified embeddings.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(text_features, video_features)
            
            text_probs = outputs["text_summ_probs"]
            video_probs = outputs["video_summ_probs"]
            T_matrix = outputs["ot_alignment_matrix"]
            
            matched_summaries = []
            B = T_matrix.shape[0]
            
            for b in range(B):
                # Top-K candidates
                k_text = min(top_k, text_probs.shape[1])
                text_topk_indices = torch.topk(text_probs[b], k_text).indices
                
                k_video = min(top_k, video_probs.shape[1])
                video_topk_indices = torch.topk(video_probs[b], k_video).indices
                
                # Pair using OT transport plan
                alignments = []
                for t_idx in text_topk_indices:
                    for v_idx in video_topk_indices:
                        match_score = T_matrix[b, t_idx, v_idx].item()
                        alignments.append({
                            "text_idx": t_idx.item(),
                            "video_idx": v_idx.item(),
                            "match_score": match_score,
                            "text_prob": text_probs[b, t_idx].item(),
                            "video_prob": video_probs[b, v_idx].item()
                        })
                
                alignments.sort(key=lambda x: x["match_score"], reverse=True)
                matched_summaries.append(alignments)
                
            return matched_summaries
