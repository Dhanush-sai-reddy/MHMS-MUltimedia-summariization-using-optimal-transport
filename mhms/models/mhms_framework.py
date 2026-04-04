import torch
import torch.nn as nn
import torch.nn.functional as F
from mhms.models.video_temporal_segmentation import VTS
from mhms.models.summarization import TextExtractiveSummarizer, VisualEncoderDecoderSummarizer

class MHMS(nn.Module):
    """
    The Multimodal Hierarchical Multimedia Summarization (MHMS) framework.
    Combines Textual Segmentation, Video Segmentation, and Extractive Summarization
    under a unified model aligned via Optimal Transport.
    """
    def __init__(self, 
                 text_hidden_size=256, 
                 visual_feature_dim=1024, 
                 video_hidden_dim=256,
                 video_omega_b=3):
        super(MHMS, self).__init__()
        
        # 1. Modality Segmenters (Feature Extractors)
        # Using a simple Bi-GRU over custom word embeddings since we have the summaries and don't need heavy BERTs!
        self.text_embedding = nn.Embedding(num_embeddings=30522, embedding_dim=128, padding_idx=0)
        self.text_gru = nn.GRU(input_size=128, hidden_size=text_hidden_size//2, bidirectional=True, batch_first=True)
        
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
        self.text_summarizer = TextExtractiveSummarizer(input_dim=text_hidden_size, hidden_dim=text_hidden_size//2)
        
        # For video summarization, we extract visual seq attention scores directly using Encoder-Decoder
        self.video_summarizer = VisualEncoderDecoderSummarizer(input_dim=visual_feature_dim, hidden_dim=video_hidden_dim)

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
        
        return distance, T

    def forward(self, input_ids, attention_mask, video_features):
        """
        Args:
            input_ids: (B, Num_Sentences, Max_Words)
            attention_mask: (B, Num_Sentences, Max_Words)
            video_features: (B, Num_Shots, Video_Dim)
            
        Returns:
            dict containing probabilities and OT loss
        """
        
        # 1. Text Feature Extraction (Lightweight GRU instead of BERT)
        # Assuming input_ids is (B, Sentences, Words), we treat as (B*Sentences, Words)
        B, S, W = input_ids.shape
        flat_input = input_ids.view(B * S, W)
        
        embeds = self.text_embedding(flat_input)     # (B*S, W, 128)
        gru_out, _ = self.text_gru(embeds)           # (B*S, W, text_hidden_size)
        
        # Mean pool over words to get sentence feature natively
        sentence_feats = gru_out.mean(dim=1)         # (B*S, text_hidden_size)
        text_features = sentence_feats.view(B, S, -1) # (B, S, text_hidden_size)
        
        # We don't need text segmentation probabilities since we aren't doing the full BERT split!
        text_seg_probs = torch.zeros(B, S, device=input_ids.device)
        
        # 2. Video Segmentation
        video_seg_probs = self.video_segmenter(video_features)
        
        # 3. Summarization
        text_summ_probs = self.text_summarizer(text_features)
        video_summ_probs = self.video_summarizer(video_features)
        
        # 4. Optimal Transport Cross-modal Alignment
        # To train the unsupervised visual summarizer, its predicted probabilities MUST 
        # modulate its features so that OT alignment gradients flow back through the decoder.
        E_weighted = text_features * text_summ_probs.unsqueeze(-1)
        V_weighted = video_features * video_summ_probs.unsqueeze(-1)
        
        # Project weighted features to Same Alignment Dimension
        E_proj = self.text_projector(E_weighted)
        V_proj = self.video_projector(V_weighted)
        
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

    def generate_multimodal_summary(self, input_ids, attention_mask, video_features, threshold=0.5):
        """
        Follows the exact inference mechanism described in the MHMS paper Section 3.5.
        Generates the visual and textual candidates, then uses Optimal Transport 
        to compute the alignment Matrix T to select the best matching multimedia pairs.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, video_features)
            
            text_probs = outputs["text_summ_probs"]   # (B, Num_Sentences)
            video_probs = outputs["video_summ_probs"] # (B, Num_Shots)
            T_matrix = outputs["ot_alignment_matrix"] # (B, Num_Sentences, Num_Shots)
            
            # The matrix T denotes the Wasserstein transportation cost matching.
            # Best matches are pairs with the highest mass transport values in T.
            matched_summaries = []
            B = T_matrix.shape[0]
            
            for b in range(B):
                # 1. Select Candidates via Summarizer probabilities
                text_candidates = torch.where(text_probs[b] > threshold)[0]
                video_candidates = torch.where(video_probs[b] > threshold)[0]
                
                # 2. Align candidates using the Transport Plan matrix T
                alignments = []
                for t_idx in text_candidates:
                    for v_idx in video_candidates:
                        # Higher value in T means stronger alignment 
                        match_score = T_matrix[b, t_idx, v_idx].item()
                        alignments.append({
                            "text_idx": t_idx.item(),
                            "video_idx": v_idx.item(),
                            "match_score": match_score
                        })
                        
                # Sort alignments to fetch the highest correlated multimodal summary pairs
                alignments.sort(key=lambda x: x["match_score"], reverse=True)
                matched_summaries.append(alignments)
                
            return matched_summaries
