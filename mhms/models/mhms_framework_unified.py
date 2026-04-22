import torch
import torch.nn as nn
import torch.nn.functional as F
from mhms.models.video_temporal_segmentation import VTS
from mhms.models.summarization import TextExtractiveSummarizer, VisualEncoderDecoderSummarizer

class MHMS_Unified(nn.Module):
    def __init__(self, embedding_dim=1536, video_hidden_dim=512, text_hidden_dim=512, video_omega_b=3):
        super(MHMS_Unified, self).__init__()
        self.embedding_dim = embedding_dim
        self.alignment_dim = 256
        self.video_segmenter = VTS(visual_feature_dim=embedding_dim, hidden_dim=video_hidden_dim, omega_b=video_omega_b)
        self.text_projector = nn.Linear(embedding_dim, self.alignment_dim)
        self.video_projector = nn.Linear(embedding_dim, self.alignment_dim)
        self.text_summarizer = TextExtractiveSummarizer(input_dim=embedding_dim, hidden_dim=text_hidden_dim)
        self.video_summarizer = VisualEncoderDecoderSummarizer(input_dim=embedding_dim, hidden_dim=video_hidden_dim)

    def compute_sinkhorn_loss_torch(self, E, V, reg=0.05, num_iters=50, text_mask=None, video_mask=None):
        B, K, D = E.shape
        _, M, _ = V.shape
        E_norm = F.normalize(E, p=2, dim=-1, eps=1e-8)
        V_norm = F.normalize(V, p=2, dim=-1, eps=1e-8)
        C = 1.0 - torch.bmm(E_norm, V_norm.transpose(1, 2))

        if text_mask is not None and video_mask is not None:
            valid_mask = text_mask.unsqueeze(2).float() * video_mask.unsqueeze(1).float()
            C = C * valid_mask + (1.0 - valid_mask) * 1e6
            mu = text_mask.float() / text_mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
            nu = video_mask.float() / video_mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
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
        return torch.sum(T * C) / B, T

    def forward(self, text_features, video_features, text_mask=None, video_mask=None):
        B, S, _ = text_features.shape
        video_seg_probs = self.video_segmenter(video_features)
        text_summ_probs = self.text_summarizer(text_features)
        video_summ_probs = self.video_summarizer(video_features)

        E_proj = self.text_projector(text_features * text_summ_probs.unsqueeze(-1))
        V_proj = self.video_projector(video_features * video_summ_probs.unsqueeze(-1))
        ot_loss, T_matrix = self.compute_sinkhorn_loss_torch(E_proj, V_proj, text_mask=text_mask, video_mask=video_mask)

        return {
            "video_seg_probs": video_seg_probs,
            "text_summ_probs": text_summ_probs,
            "video_summ_probs": video_summ_probs,
            "ot_loss": ot_loss,
            "ot_alignment_matrix": T_matrix
        }

    def generate_multimodal_summary_topk(self, text_features, video_features, top_k=3, text_mask=None, video_mask=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(text_features, video_features, text_mask=text_mask, video_mask=video_mask)
            text_probs = outputs["text_summ_probs"]
            video_probs = outputs["video_summ_probs"]
            T_matrix = outputs["ot_alignment_matrix"]

            matched_summaries = []
            for b in range(T_matrix.shape[0]):
                n_vt = int(text_mask[b].sum().item()) if text_mask is not None else text_probs.shape[1]
                n_vv = int(video_mask[b].sum().item()) if video_mask is not None else video_probs.shape[1]
                text_topk = torch.topk(text_probs[b, :n_vt], min(top_k, n_vt)).indices
                video_topk = torch.topk(video_probs[b, :n_vv], min(top_k, n_vv)).indices
                alignments = [
                    {"text_idx": ti.item(), "video_idx": vi.item(),
                     "match_score": T_matrix[b, ti, vi].item(),
                     "text_prob": text_probs[b, ti].item(),
                     "video_prob": video_probs[b, vi].item()}
                    for ti in text_topk for vi in video_topk
                ]
                alignments.sort(key=lambda x: x["match_score"], reverse=True)
                matched_summaries.append(alignments)
            return matched_summaries
