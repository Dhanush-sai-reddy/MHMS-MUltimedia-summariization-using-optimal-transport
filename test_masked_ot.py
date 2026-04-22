"""Quick smoke test: masked OT vs unmasked OT."""
import torch
from mhms.models.mhms_framework import MHMS

m = MHMS()
t = torch.randn(1, 20, 768)   # 20 padded slots
v = torch.randn(1, 20, 2048)
# Only 8 real sentences, 5 real frames
tm = torch.zeros(1, 20, dtype=torch.long)
vm = torch.zeros(1, 20, dtype=torch.long)
tm[0, :8] = 1
vm[0, :5] = 1
# Zero out padded positions (simulating real padding)
t[0, 8:] = 0
v[0, 5:] = 0

# With masks
o = m(t, v, text_mask=tm, video_mask=vm)
T = o["ot_alignment_matrix"][0]
valid_mass = T[:8, :5].sum().item()
pad_mass = T.sum().item() - valid_mass

# Without masks (old behavior)
o2 = m(t, v)
T2 = o2["ot_alignment_matrix"][0]
valid_mass2 = T2[:8, :5].sum().item()
pad_mass2 = T2.sum().item() - valid_mass2

print(f"MASKED:   valid={valid_mass:.4f}  padding={pad_mass:.6f}  OT_loss={o['ot_loss'].item():.4f}")
print(f"UNMASKED: valid={valid_mass2:.4f}  padding={pad_mass2:.6f}  OT_loss={o2['ot_loss'].item():.4f}")
print(f"Padding mass reduced by {pad_mass2/max(pad_mass,1e-10):.0f}x")
