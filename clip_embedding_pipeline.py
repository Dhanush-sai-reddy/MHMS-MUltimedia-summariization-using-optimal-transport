"""
MHMS CLIP Embedding Pipeline
=============================
Extracts joint multimodal embeddings using OpenAI CLIP (ViT-B/32).
Both text and visual features live in the SAME 512-dim space,
which is ideal for Optimal Transport alignment.

- Text:   CLIP text encoder  -> 512-dim per sentence
- Visual: CLIP vision encoder -> 512-dim per keyframe

Output structure:
  embeddings_clip/
    text/
      case_1.npy, case_2.npy, ...   (N x 512)
    visual/
      case_1.npy, case_2.npy, ...   (M x 512)
    manifest.json
"""

import os
import sys
import json
import glob
import torch
import numpy as np
from PIL import Image

try:
    import clip
except ImportError:
    print("ERROR: 'clip' package not found. Install it with:")
    print("  pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)


# ─── 1. INITIALIZE CLIP ─────────────────────────────────────────────

CLIP_MODEL_NAME = "ViT-B/32"       # 512-dim embeddings
EMBEDDING_DIM   = 512

print(f"Loading CLIP model '{CLIP_MODEL_NAME}' ...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
model.eval()
print("CLIP model loaded successfully.\n")


# ─── 2. EXTRACTION HELPERS ──────────────────────────────────────────

def extract_text_embeddings(sentences, batch_size=64):
    """
    Encode sentences with CLIP's text encoder.
    Returns (N, 512) float32 numpy array.
    """
    if not sentences:
        return np.array([], dtype=np.float32)

    all_features = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        # CLIP tokenizer truncates at 77 tokens internally
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)           # (B, 512) float16/32
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalize
            all_features.append(feats.cpu().float().numpy())

    return np.concatenate(all_features, axis=0).astype(np.float32)


def extract_visual_embeddings(image_paths, batch_size=32):
    """
    Encode keyframe images with CLIP's vision encoder.
    Returns (M, 512) float32 numpy array.
    """
    if not image_paths:
        return np.array([], dtype=np.float32)

    all_features = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                print(f"  ⚠ Skipping {p}: {e}")
                # Use a blank tensor so indices stay aligned
                images.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)    # (B, 512)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu().float().numpy())

    return np.concatenate(all_features, axis=0).astype(np.float32)


# ─── 3. MAIN PIPELINE ───────────────────────────────────────────────

def main():
    data_dir = "cnn_data"
    out_dir  = "embeddings_clip"

    if not os.path.exists(data_dir):
        print(f"Error: '{data_dir}' not found.")
        sys.exit(1)

    text_dir   = os.path.join(out_dir, "text")
    visual_dir = os.path.join(out_dir, "visual")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    print("=" * 60)
    print("  MHMS CLIP Embedding Pipeline")
    print(f"  Model: {CLIP_MODEL_NAME}  |  Dim: {EMBEDDING_DIM}")
    print("=" * 60)

    # Discover case directories
    case_dirs = []
    for d in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, d)
        if os.path.isdir(dir_path) and d.isdigit():
            case_dirs.append((int(d), dir_path))
    case_dirs.sort()

    text_count   = 0
    visual_count = 0
    skipped      = 0
    manifest     = {"embedding_dim": EMBEDDING_DIM, "model": CLIP_MODEL_NAME, "cases": {}}

    for case_id, case_path in case_dirs:
        entry = {"case_id": case_id}

        # ── Text embeddings ──
        text_file = os.path.join(case_path, "artitle_section.txt")
        text_emb  = None
        if os.path.isfile(text_file):
            with open(text_file, "r", encoding="utf-8") as f:
                sentences = [s.strip() for s in f.readlines() if s.strip()]
            if sentences:
                text_emb = extract_text_embeddings(sentences)
                np.save(os.path.join(text_dir, f"case_{case_id}.npy"), text_emb)
                entry["text_shape"] = list(text_emb.shape)
                text_count += 1

        # ── Visual embeddings ──
        keyframes = sorted(glob.glob(os.path.join(case_path, "*_summary.jpg")))
        visual_emb = None
        if keyframes:
            visual_emb = extract_visual_embeddings(keyframes)
            np.save(os.path.join(visual_dir, f"case_{case_id}.npy"), visual_emb)
            entry["visual_shape"] = list(visual_emb.shape)
            entry["keyframes"]    = [os.path.basename(p) for p in keyframes]
            visual_count += 1

        if text_emb is not None or visual_emb is not None:
            manifest["cases"][str(case_id)] = entry
            # Print progress for first few + every 50th
            if (text_count + visual_count) <= 10 or case_id % 50 == 0:
                t = str(list(text_emb.shape))  if text_emb  is not None else "–"
                v = str(list(visual_emb.shape)) if visual_emb is not None else "–"
                print(f"  Case {case_id:>3d}  |  text: {t:>12s}  |  visual: {v:>12s}")
        else:
            skipped += 1

    # Save manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Extraction Complete!")
    print(f"  Text embeddings:    {text_count:>3d} cases  ({EMBEDDING_DIM}-dim CLIP)")
    print(f"  Visual embeddings:  {visual_count:>3d} cases  ({EMBEDDING_DIM}-dim CLIP)")
    print(f"  Skipped:            {skipped:>3d} cases")
    print(f"  Output directory:   {out_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
