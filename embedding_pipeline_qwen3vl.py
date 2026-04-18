"""
MHMS Embedding Pipeline (Qwen3 VL Unified Vision-Language)
===========================================================
Extracts unified multimodal embeddings using Qwen3 VL:
- Both text and images encoded into the SAME semantic space
- Native vision-language understanding for better cross-modal alignment
- Compatible with the Optimal Transport alignment in MHMS

Output structure:
  embeddings_qwen3vl/
    text/       - case_1.npy, case_2.npy, ... (N x 3584) - sentence embeddings
    visual/     - case_1.npy, case_2.npy, ... (M x 3584) - keyframe embeddings
    manifest.json

Requirements:
  pip install transformers accelerate qwen-vl-utils

Model: Qwen/Qwen3-VL-8B (or Qwen3-VL-2B for lighter memory)
  - Hidden dimension: 3584 (8B) or 2048 (2B)
  - Supports both text and image inputs natively
"""

import os
import sys
import json
import glob
import torch
import numpy as np
from PIL import Image
from typing import List, Union

# Qwen3 VL from HuggingFace
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ─── 1. INITIALIZE MODEL ──────────────────────────────────────────────

print("Loading Qwen3-VL-4B-Instruct model (~9GB weights, may take a few minutes to load)...")

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model ID - Qwen3-VL-4B-Instruct (smallest available, ~9GB)
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"  # or "Qwen/Qwen3-VL-8B-Instruct" for higher quality

# Load model and processor
# Note: Use float16 for efficiency on GPU, float32 on CPU (avoid bfloat16 on CPU)
model_kwargs = {"trust_remote_code": True}

if torch.cuda.is_available():
    # GPU: use float16 for efficiency
    model_kwargs["torch_dtype"] = torch.float16
    model_kwargs["device_map"] = "auto"
else:
    # CPU: load in float32 but with low_cpu_mem_usage
    model_kwargs["torch_dtype"] = torch.float32
    model_kwargs["low_cpu_mem_usage"] = True

model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Get embedding dimension from model config
# Qwen3-VL-4B-Instruct uses 2560 dim (text config) or 2048 (vision config)
try:
    EMBEDDING_DIM = model.config.text_config.hidden_size  # 2560 for 4B
except AttributeError:
    EMBEDDING_DIM = getattr(model.config, 'hidden_size', 2560)
print(f"Model loaded. Embedding dimension: {EMBEDDING_DIM}")

model.eval()


# ─── 2. EXTRACTION LOGIC ──────────────────────────────────────────────

def extract_text_embeddings_qwen(sentences: List[str], batch_size: int = 8) -> np.ndarray:
    """
    Encode sentences using Qwen3 VL text encoder.
    Extracts hidden states from the last layer as embeddings.
    
    Args:
        sentences: List of text sentences
        batch_size: Number of sentences to process at once
        
    Returns:
        embeddings: (N, EMBEDDING_DIM) numpy array
    """
    if not sentences:
        return np.array([])
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Format for Qwen3 VL: each input is a conversation with text
            conversations = []
            for sent in batch:
                conversations.append([
                    {"role": "user", "content": [{"type": "text", "text": sent}]}
                ])
            
            # Process inputs
            text_inputs = processor(
                text=[processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) 
                      for conv in conversations],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # Get hidden states from the model
            outputs = model(**text_inputs, output_hidden_states=True, return_dict=True)
            
            # Extract embeddings from last hidden state (average pooling over tokens)
            # Shape: (batch, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states[-1]
            
            # Mean pooling over valid tokens (excluding padding)
            attention_mask = text_inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
    
    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def extract_visual_embeddings_qwen(image_paths: List[str], batch_size: int = 4) -> np.ndarray:
    """
    Encode images using Qwen3 VL vision encoder.
    Extracts hidden states from the last layer as embeddings.
    
    Args:
        image_paths: List of image file paths
        batch_size: Number of images to process at once (keep low for memory)
        
    Returns:
        embeddings: (M, EMBEDDING_DIM) numpy array
    """
    if not image_paths:
        return np.array([])
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Format for Qwen3 VL: each input is a conversation with image
            conversations = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    # Verify image can be opened
                    with Image.open(path) as img:
                        img.convert('RGB')
                    
                    conversations.append([
                        {"role": "user", "content": [
                            {"type": "image", "image": path},
                            {"type": "text", "text": "Describe this image."}  # Dummy text prompt
                        ]}
                    ])
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Warning: Could not load image {path}: {e}")
                    # Will append zero embedding later
            
            if not conversations:
                # All images in batch failed
                for _ in batch_paths:
                    all_embeddings.append(np.zeros((1, EMBEDDING_DIM), dtype=np.float32))
                continue
            
            try:
                # Process inputs
                text_inputs = processor(
                    text=[processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) 
                          for conv in conversations],
                    images=process_vision_info(conversations)[0],
                    return_tensors="pt",
                    padding=True
                ).to(model.device)
                
                # Get hidden states from the model
                outputs = model(**text_inputs, output_hidden_states=True, return_dict=True)
                
                # Extract embeddings from last hidden state
                # For vision inputs, we average pool the relevant vision tokens
                hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
                
                # Use mean pooling over all tokens (vision + text)
                attention_mask = text_inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                batch_embeddings = embeddings.cpu().to(torch.float32).numpy()
                
                # Add embeddings for valid images
                for j, emb in enumerate(batch_embeddings):
                    all_embeddings.append(emb.reshape(1, -1))
                
                # Add zero embeddings for failed images
                failed_count = len(batch_paths) - len(valid_paths)
                for _ in range(failed_count):
                    all_embeddings.append(np.zeros((1, EMBEDDING_DIM), dtype=np.float32))
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add zero embeddings for entire batch
                for _ in batch_paths:
                    all_embeddings.append(np.zeros((1, EMBEDDING_DIM), dtype=np.float32))
    
    if not all_embeddings:
        return np.array([])
    
    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


# ─── 3. MAIN PIPELINE ────────────────────────────────────────────────

def main():
    data_dir = "cnn_data"
    out_dir = "embeddings_qwen3vl"

    if not os.path.exists(data_dir):
        print(f"Error: '{data_dir}' not found.")
        sys.exit(1)

    text_dir = os.path.join(out_dir, "text")
    visual_dir = os.path.join(out_dir, "visual")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  MHMS Embedding Pipeline (Qwen3 VL Unified)")
    print(f"  Model: {MODEL_ID}")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
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
    manifest = {"model": MODEL_ID, "embedding_dim": EMBEDDING_DIM, "cases": {}}

    for case_id, case_path in case_dirs:
        entry = {"case_id": case_id}

        # ── Text embeddings (Qwen3 VL) ──
        text_file = os.path.join(case_path, "artitle_section.txt")
        text_emb = None
        if os.path.isfile(text_file):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    sentences = [s.strip() for s in f.readlines() if s.strip()]
                if sentences:
                    print(f"  Processing text case {case_id} ({len(sentences)} sentences)...")
                    text_emb = extract_text_embeddings_qwen(sentences)
                    save_path = os.path.join(text_dir, f"case_{case_id}.npy")
                    np.save(save_path, text_emb)
                    entry["text_shape"] = list(text_emb.shape)
                    text_count += 1
            except Exception as e:
                print(f"  Error processing text for case {case_id}: {e}")

        # ── Visual embeddings (Qwen3 VL) ──
        keyframes = sorted(glob.glob(os.path.join(case_path, "*_summary.jpg")))
        visual_emb = None
        if keyframes:
            try:
                print(f"  Processing visual case {case_id} ({len(keyframes)} keyframes)...")
                visual_emb = extract_visual_embeddings_qwen(keyframes)
                save_path = os.path.join(visual_dir, f"case_{case_id}.npy")
                np.save(save_path, visual_emb)
                entry["visual_shape"] = list(visual_emb.shape)
                entry["keyframes"] = [os.path.basename(p) for p in keyframes]
                visual_count += 1
            except Exception as e:
                print(f"  Error processing visual for case {case_id}: {e}")

        if text_emb is not None or visual_emb is not None:
            manifest["cases"][str(case_id)] = entry
            t = str(list(text_emb.shape)) if text_emb is not None else "–"
            v = str(list(visual_emb.shape)) if visual_emb is not None else "–"
            print(f"  ✓ Case {case_id:>3d}  |  text: {t:>12s}  |  visual: {v:>12s}")
        else:
            skipped += 1

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Extraction Complete!")
    print(f"  Model:              {MODEL_ID}")
    print(f"  Embedding dim:      {EMBEDDING_DIM}")
    print(f"  Text embeddings:    {text_count:>3d} cases")
    print(f"  Visual embeddings:  {visual_count:>3d} cases")
    print(f"  Skipped:            {skipped:>3d} cases")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"  1. Update dataset.py to use embeddings_qwen3vl/")
    print(f"  2. Update mhms_framework.py dims: text={EMBEDDING_DIM}, visual={EMBEDDING_DIM}")
    print(f"  3. Retrain model with unified embeddings")


if __name__ == "__main__":
    main()
