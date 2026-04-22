import torch
import torch.nn as nn
import torch.nn.functional as F
from mhms.models.video_temporal_segmentation import VTS
from mhms.models.text_segmentation import HierarchicalBERT
from mhms.models.summarization import TextExtractiveSummarizer, VisualEncoderDecoderSummarizer

class MHMS(nn.Module):
    def __init__(self, text_feature_dim=768, visual_feature_dim=2048, video_hidden_dim=256, video_omega_b=3):
        super(MHMS, self).__init__()
        self.text_segmenter = HierarchicalBERT(
            pretrained_model_name='bert-base-uncased',
            num_article_layers=12, hidden_size=768, num_heads=12
        )
        self.video_segmenter = VTS(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=video_hidden_dim, omega_b=video_omega_b
        )
        self.alignment_dim = 256
        self.text_projector = nn.Linear(text_feature_dim, self.alignment_dim)
        self.video_projector = nn.Linear(visual_feature_dim, self.alignment_dim)
        self.text_summarizer = TextExtractiveSummarizer(input_dim=text_feature_dim, hidden_dim=text_feature_dim//2)
        self.video_summarizer = VisualEncoderDecoderSummarizer(input_dim=visual_feature_dim, hidden_dim=video_hidden_dim)

    def compute_sinkhorn_loss_torch(self, E, V, reg=0.05, num_iters=50, text_mask=None, video_mask=None):
        B, K, D = E.shape
        _, M, _ = V.shape

        E_norm = F.normalize(E, p=2, dim=-1, eps=1e-8)
        V_norm = F.normalize(V, p=2, dim=-1, eps=1e-8)
        sim = torch.bmm(E_norm, V_norm.transpose(1, 2))
        C = 1.0 - sim

        if text_mask is not None and video_mask is not None:
            valid_mask = text_mask.unsqueeze(2).float() * video_mask.unsqueeze(1).float()
            C = C * valid_mask + (1.0 - valid_mask) * 1e6
            text_counts = text_mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
            video_counts = video_mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
            mu = text_mask.float() / text_counts
            nu = video_mask.float() / video_counts
        else:
            mu = torch.ones(B, K, device=E.device) / K
            nu = torch.ones(B, M, device=E.device) / M

        K_matrix = torch.exp(-C / reg)
        u = torch.ones(B, K, device=E.device) / K
        v = torch.ones(B, M, device=E.device) / M

        for _ in range(num_iters):
            u = mu / (torch.bmm(K_matrix, v.unsqueeze(2)).squeeze(2) + 1e-8)
            v = nu / (torch.bmm(K_matrix.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)

        T = u.unsqueeze(2) * K_matrix * v.unsqueeze(1)
        distance = torch.sum(T * C) / B
        return distance, T

    def forward(self, text_features, video_features, text_input_ids=None, text_attention_mask=None, text_mask=None, video_mask=None):
        B, S, _ = text_features.shape

        if text_input_ids is not None and text_attention_mask is not None:
            text_seg_probs, _ = self.text_segmenter(text_input_ids, text_attention_mask)
        else:
            text_seg_probs = torch.zeros(B, S, device=text_features.device)

        video_seg_probs = self.video_segmenter(video_features)
        text_summ_probs = self.text_summarizer(text_features)
        video_summ_probs = self.video_summarizer(video_features)

        E_weighted = text_features * text_summ_probs.unsqueeze(-1)
        V_weighted = video_features * video_summ_probs.unsqueeze(-1)
        E_proj = self.text_projector(E_weighted)
        V_proj = self.video_projector(V_weighted)

        ot_loss, T_matrix = self.compute_sinkhorn_loss_torch(E_proj, V_proj, text_mask=text_mask, video_mask=video_mask)

        return {
            "text_seg_probs": text_seg_probs,
            "text_features": text_features,
            "video_seg_probs": video_seg_probs,
            "text_summ_probs": text_summ_probs,
            "video_summ_probs": video_summ_probs,
            "ot_loss": ot_loss,
            "ot_alignment_matrix": T_matrix
        }

    def generate_multimodal_summary(self, text_features, video_features, threshold=0.5, text_mask=None, video_mask=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(text_features, video_features, text_mask=text_mask, video_mask=video_mask)
            text_probs = outputs["text_summ_probs"]
            video_probs = outputs["video_summ_probs"]
            T_matrix = outputs["ot_alignment_matrix"]

            matched_summaries = []
            for b in range(T_matrix.shape[0]):
                text_candidates = torch.where(text_probs[b] > threshold)[0]
                video_candidates = torch.where(video_probs[b] > threshold)[0]
                alignments = []
                for t_idx in text_candidates:
                    for v_idx in video_candidates:
                        alignments.append({
                            "text_idx": t_idx.item(),
                            "video_idx": v_idx.item(),
                            "match_score": T_matrix[b, t_idx, v_idx].item()
                        })
                alignments.sort(key=lambda x: x["match_score"], reverse=True)
                matched_summaries.append(alignments)
            return matched_summaries

    def generate_multimodal_summary_topk(self, text_features, video_features, top_k=3, text_mask=None, video_mask=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(text_features, video_features, text_mask=text_mask, video_mask=video_mask)
            text_probs = outputs["text_summ_probs"]
            video_probs = outputs["video_summ_probs"]
            T_matrix = outputs["ot_alignment_matrix"]

            matched_summaries = []
            for b in range(T_matrix.shape[0]):
                n_valid_text = int(text_mask[b].sum().item()) if text_mask is not None else text_probs.shape[1]
                n_valid_video = int(video_mask[b].sum().item()) if video_mask is not None else video_probs.shape[1]

                k_text = min(top_k, n_valid_text)
                k_video = min(top_k, n_valid_video)
                text_topk = torch.topk(text_probs[b, :n_valid_text], k_text).indices
                video_topk = torch.topk(video_probs[b, :n_valid_video], k_video).indices

                alignments = []
                for t_idx in text_topk:
                    for v_idx in video_topk:
                        alignments.append({
                            "text_idx": t_idx.item(),
                            "video_idx": v_idx.item(),
                            "match_score": T_matrix[b, t_idx, v_idx].item(),
                            "text_prob": text_probs[b, t_idx].item(),
                            "video_prob": video_probs[b, v_idx].item()
                        })
                alignments.sort(key=lambda x: x["match_score"], reverse=True)
                matched_summaries.append(alignments)
            return matched_summaries
