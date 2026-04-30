"""Quick smoke test: dataset + one forward pass with CLIP embeddings."""
import torch
from mhms.dataset_unified import CNNMultimodalDatasetUnified
from mhms.models.mhms_framework_unified import MHMS_Unified
from torch.utils.data import DataLoader

ds = CNNMultimodalDatasetUnified(embeddings_dir="embeddings_clip", embedding_dim=512)
print(f"Total samples: {len(ds)}")

dl = DataLoader(ds, batch_size=4, shuffle=True)
batch = next(iter(dl))
print(f"text_features:  {batch['text_features'].shape}")
print(f"video_features: {batch['video_features'].shape}")

model = MHMS_Unified(embedding_dim=512, video_hidden_dim=512, text_hidden_dim=512, video_omega_b=3)
out = model(batch["text_features"], batch["video_features"],
            text_mask=batch["text_mask"], video_mask=batch["video_mask"])
print(f"ot_loss: {out['ot_loss'].item():.4f}")
print("Forward pass OK")
