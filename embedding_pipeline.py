"""
MHMS Embedding Pipeline
========================
Extracts text and visual embeddings from the CNN dataset
and stores them centrally in the embeddings/ directory.

- Text Embeddings:   TF-IDF vectors from article sentences
- Visual Embeddings: Color histogram + texture features from keyframe images

Output structure:
  embeddings/
    text/
      case_1.npy, case_2.npy, ...
    visual/
      case_1.npy, case_2.npy, ...
    vectorizer.npy   (fitted TF-IDF vocabulary for reuse)

These embeddings are consumed by optimal_transport.py for cross-modal alignment.
"""

import os
import sys
import json
import pickle
import numpy as np
import cv2
import glob
from sklearn.feature_extraction.text import TfidfVectorizer


# ─── Text Embeddings ─────────────────────────────────────────────────

def build_corpus_vectorizer(data_dir, max_features=512):
    """
    Fit a global TF-IDF vectorizer on all articles for consistent dimensionality.
    """
    all_sentences = []
    for d in sorted(os.listdir(data_dir)):
        text_file = os.path.join(data_dir, d, "artitle_section.txt")
        if os.path.isfile(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                sents = [s.strip() for s in f.readlines() if s.strip()]
                all_sentences.extend(sents)

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    vectorizer.fit(all_sentences)
    return vectorizer


def extract_text_embeddings(sentences, vectorizer):
    """
    Encode sentences into TF-IDF vectors.
    Returns: (N_sentences, D) numpy array
    """
    return vectorizer.transform(sentences).toarray().astype(np.float32)


# ─── Visual Embeddings ───────────────────────────────────────────────

def extract_visual_embedding(image_path):
    """
    Extract a feature vector from a single keyframe image.
      - HSV color histogram (8x8x8 = 512 dim)
      - Grayscale spatial texture (8x8 = 64 dim)
    Total: 576-dim, L2-normalized.
    """
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(576, dtype=np.float32)

    # Color histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Spatial texture
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    spatial = cv2.resize(gray, (8, 8)).flatten().astype(np.float32)
    spatial = spatial / (np.linalg.norm(spatial) + 1e-8)

    feature = np.concatenate([hist, spatial]).astype(np.float32)
    feature = feature / (np.linalg.norm(feature) + 1e-8)
    return feature


def extract_visual_embeddings(image_paths):
    """
    Extract visual embeddings for a list of keyframe images.
    Returns: (N_images, 576) numpy array
    """
    return np.array([extract_visual_embedding(p) for p in image_paths], dtype=np.float32)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    data_dir = "cnn_data"
    out_dir = "embeddings"

    if not os.path.exists(data_dir):
        print(f"Error: '{data_dir}' not found.")
        sys.exit(1)

    # Create output directories
    text_dir = os.path.join(out_dir, "text")
    visual_dir = os.path.join(out_dir, "visual")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    print("=" * 60)
    print("  MHMS Embedding Pipeline")
    print("=" * 60)

    # Step 1: Fit global TF-IDF vectorizer
    print("\n[1/3] Fitting global TF-IDF vectorizer...")
    vectorizer = build_corpus_vectorizer(data_dir)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"  Vocabulary: {vocab_size} features")

    # Save vectorizer for reuse
    vec_path = os.path.join(out_dir, "tfidf_vectorizer.pkl")
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved → {vec_path}")

    # Step 2: Extract embeddings per case
    print("\n[2/3] Extracting embeddings...\n")

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

        # ── Text embeddings ──
        text_file = os.path.join(case_path, "artitle_section.txt")
        if os.path.isfile(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                sentences = [s.strip() for s in f.readlines() if s.strip()]
            if sentences:
                text_emb = extract_text_embeddings(sentences, vectorizer)
                save_path = os.path.join(text_dir, f"case_{case_id}.npy")
                np.save(save_path, text_emb)
                entry["text_shape"] = list(text_emb.shape)
                text_count += 1

        # ── Visual embeddings ──
        keyframes = sorted(glob.glob(os.path.join(case_path, "*_summary.jpg")))
        if keyframes:
            visual_emb = extract_visual_embeddings(keyframes)
            save_path = os.path.join(visual_dir, f"case_{case_id}.npy")
            np.save(save_path, visual_emb)
            entry["visual_shape"] = list(visual_emb.shape)
            entry["keyframes"] = [os.path.basename(p) for p in keyframes]
            visual_count += 1

        if "text_shape" in entry or "visual_shape" in entry:
            manifest[str(case_id)] = entry
            if text_count + visual_count <= 10 or case_id % 50 == 0:
                t = entry.get("text_shape", "–")
                v = entry.get("visual_shape", "–")
                print(f"  Case {case_id:>3d}  |  text: {str(t):>12s}  |  visual: {str(v):>12s}")
        else:
            skipped += 1

    # Save manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Step 3: Summary
    print(f"\n[3/3] Complete!")
    print(f"\n{'=' * 60}")
    print(f"  Text embeddings:    {text_count:>3d} cases → embeddings/text/")
    print(f"  Visual embeddings:  {visual_count:>3d} cases → embeddings/visual/")
    print(f"  Skipped:            {skipped:>3d} cases")
    print(f"  Manifest:           {manifest_path}")
    print(f"  Vectorizer:         {vec_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
