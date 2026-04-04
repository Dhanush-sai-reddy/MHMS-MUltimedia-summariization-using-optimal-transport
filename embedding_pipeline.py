"""
MHMS Embedding Pipeline (Strict Paper Implementation)
===================================================
Extracts deep semantic embeddings exactly as described in the MHMS paper:
- Text:   BERT (bert-base-uncased) -> 768-dim per sentence
- Visual: ResNet (resnet50) -> 2048-dim per keyframe

Output structure:
  embeddings/
    text/
      case_1.npy, case_2.npy, ... (N x 768)
    visual/
      case_1.npy, case_2.npy, ... (M x 2048)
    manifest.json
"""

import os
import sys
import json
import glob
import torch
import numpy as np
from PIL import Image

# HuggingFace Transformers for Text
from transformers import BertTokenizer, BertModel

# Torchvision for Images
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights


# ─── 1. INITIALIZE MODELS ───────────────────────────────────────────

print("Initializing deep learning models (this may download weights)...")

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# A. Text Model: BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = BertModel.from_pretrained('bert-base-uncased').to(device)
text_model.eval()

# B. Visual Model: ResNet50
# We want the feature map before the final classification FC layer (2048-dim)
vis_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
# Remove the final classification layer
vis_model = torch.nn.Sequential(*list(vis_model.children())[:-1])
vis_model.eval()

# Standard ImageNet Transfoms
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


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
    Encode a list of images using ResNet50.
    Returns: (M, 2048) float32 numpy array.
    """
    if not image_paths:
        return np.array([])
        
    features = []
    
    with torch.no_grad():
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                tensor = img_transform(img).unsqueeze(0).to(device)
                
                # Output shape from sequential ResNet is (1, 2048, 1, 1)
                feat = vis_model(tensor)
                feat = feat.squeeze(-1).squeeze(-1) # (1, 2048)
                features.append(feat.cpu().numpy())
            except Exception as e:
                print(f"Error processing {path}: {e}")
                features.append(np.zeros((1, 2048), dtype=np.float32))
                
    if not features:
        return np.array([])
        
    embeddings = np.concatenate(features, axis=0)
    return embeddings


# ─── 3. MAIN PIPELINE ────────────────────────────────────────────────

def main():
    data_dir = "cnn_data"
    out_dir = "embeddings"

    if not os.path.exists(data_dir):
        print(f"Error: '{data_dir}' not found.")
        sys.exit(1)

    text_dir = os.path.join(out_dir, "text")
    visual_dir = os.path.join(out_dir, "visual")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    print("\n=" * 60)
    print("  MHMS Embedding Pipeline (BERT + ResNet)")
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
    manifest = {}

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

        # ── Visual embeddings (ResNet) ──
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
            manifest[str(case_id)] = entry
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
    print(f"  Visual embeddings:  {visual_count:>3d} cases (2048-dim ResNet)")
    print(f"  Skipped:            {skipped:>3d} cases")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
