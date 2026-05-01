"""
MHMS ViT Embedding Pipeline
===========================
Extracts deep semantic embeddings:
- Text:   BERT (bert-base-uncased) -> 768-dim per sentence
- Visual: ViT (google/vit-base-patch16-224-in21k) -> 768-dim per keyframe

Output structure:
  embeddings_vit/
    text/
      case_1.npy, case_2.npy, ... (N x 768)
    visual/
      case_1.npy, case_2.npy, ... (M x 768)
    manifest.json
"""

import os
import sys
import json
import glob
import torch
import numpy as np
from PIL import Image

# HuggingFace Transformers
from transformers import BertTokenizer, BertModel, ViTImageProcessor, ViTModel

# ─── 1. INITIALIZE MODELS ───────────────────────────────────────────

print("Initializing deep learning models (this may download weights)...")

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# A. Text Model: BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = BertModel.from_pretrained('bert-base-uncased').to(device)
text_model.eval()

# B. Visual Model: ViT
vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
vis_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
vis_model.eval()


# ─── 2. EXTRACTION LOGIC ─────────────────────────────────────────────

def extract_text_embeddings(sentences):
    """
    Encode a batch of sentences using BERT.
    Uses the [CLS] token as the sentence representation (768-dim).
    """
    if not sentences:
        return np.array([])
        
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = text_model(**inputs)
        # pooler_output is the [CLS] token representation passed through a Linear+Tanh layer
        cls_embeddings = outputs.pooler_output.cpu().numpy()
        
    return cls_embeddings.astype(np.float32)


def extract_visual_embeddings(image_paths):
    """
    Encode a list of images using ViT.
    Returns: (M, 768) float32 numpy array.
    """
    if not image_paths:
        return np.array([])
        
    features = []
    
    with torch.no_grad():
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                inputs = vit_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = vis_model(**inputs)
                # pooler_output is (1, 768)
                feat = outputs.pooler_output
                features.append(feat.cpu().numpy())
            except Exception as e:
                print(f"Error processing {path}: {e}")
                features.append(np.zeros((1, 768), dtype=np.float32))
                
    if not features:
        return np.array([])
        
    embeddings = np.concatenate(features, axis=0)
    return embeddings


# ─── 3. MAIN PIPELINE ────────────────────────────────────────────────

def main():
    data_dir = "cnn_data"
    out_dir = "embeddings_vit"

    if not os.path.exists(data_dir):
        print(f"Error: '{data_dir}' not found.")
        sys.exit(1)

    text_dir = os.path.join(out_dir, "text")
    visual_dir = os.path.join(out_dir, "visual")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    print("\n=" * 60)
    print("  MHMS Embedding Pipeline (BERT + ViT)")
    print("=" * 60)

    case_dirs = []
    for d in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, d)
        if os.path.isdir(dir_path) and d.isdigit():
            case_dirs.append((int(d), dir_path))
    case_dirs.sort()

    text_count = 0
    visual_count = 0
    skipped = 0
    manifest = {"embedding_dim": 768, "model": "vit", "cases": {}}

    for case_id, case_path in case_dirs:
        entry = {"case_id": case_id}

        # ── Text embeddings (BERT) ──
        text_file = os.path.join(case_path, "artitle_section.txt")
        text_emb = None
        if os.path.isfile(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                sentences = [s.strip() for s in f.readlines() if s.strip()]
            if sentences:
                text_emb = extract_text_embeddings(sentences)
                save_path = os.path.join(text_dir, f"case_{case_id}.npy")
                np.save(save_path, text_emb)
                entry["text_shape"] = list(text_emb.shape)
                text_count += 1

        # ── Visual embeddings (ViT) ──
        keyframes = sorted(glob.glob(os.path.join(case_path, "*_summary.jpg")))
        visual_emb = None
        if keyframes:
            visual_emb = extract_visual_embeddings(keyframes)
            save_path = os.path.join(visual_dir, f"case_{case_id}.npy")
            np.save(save_path, visual_emb)
            entry["visual_shape"] = list(visual_emb.shape)
            entry["keyframes"] = [os.path.basename(p) for p in keyframes]
            visual_count += 1

        if text_emb is not None or visual_emb is not None:
            manifest["cases"][str(case_id)] = entry
            if text_count + visual_count <= 10 or case_id % 50 == 0:
                t = str(list(text_emb.shape)) if text_emb is not None else "–"
                v = str(list(visual_emb.shape)) if visual_emb is not None else "–"
                print(f"  Case {case_id:>3d}  |  text: {t:>12s}  |  visual: {v:>12s}")
        else:
            skipped += 1

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Extraction Complete!")
    print(f"  Text embeddings:    {text_count:>3d} cases (768-dim BERT)")
    print(f"  Visual embeddings:  {visual_count:>3d} cases (768-dim ViT)")
    print(f"  Skipped:            {skipped:>3d} cases")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
