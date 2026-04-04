"""
MHMS Embeddings Pipeline
=========================
Extracts text and visual embeddings from the CNN dataset.

- Text Embeddings:   TF-IDF vectors from article sentences
- Visual Embeddings: Color histogram + texture features from keyframe images

Embeddings are saved as .npy files inside each case folder.
These are then used by optimal_transport.py for cross-modal alignment.
"""

import os
import sys
import numpy as np
import cv2
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


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
    Encode sentences into TF-IDF vectors using the global vectorizer.
    Returns: (N, D) numpy array
    """
    return vectorizer.transform(sentences).toarray().astype(np.float32)


# ─── Visual Embeddings ───────────────────────────────────────────────

def extract_visual_embedding(image_path):
    """
    Extract a feature vector from a single keyframe image.
      - HSV color histogram (8×8×8 = 512 dim)
      - Grayscale spatial texture (8×8 = 64 dim)
    Total: 576-dim vector, L2-normalized.
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
    Extract visual embeddings for a list of keyframe image paths.
    Returns: (N, 576) numpy array
    """
    embeddings = [extract_visual_embedding(p) for p in image_paths]
    return np.array(embeddings, dtype=np.float32)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    data_dir = "cnn_data"

    if not os.path.exists(data_dir):
        print(f"Error: '{data_dir}' not found.")
        sys.exit(1)

    print("=" * 60)
    print("  MHMS Embeddings Pipeline")
    print("=" * 60)

    # Step 1: Fit global TF-IDF vectorizer
    print("\n[1/3] Fitting global TF-IDF vectorizer on corpus...")
    vectorizer = build_corpus_vectorizer(data_dir)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)} features")

    # Step 2: Process each case folder
    print("\n[2/3] Extracting embeddings per case...\n")

    case_dirs = []
    for d in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, d)
        if os.path.isdir(dir_path) and d.isdigit():
            case_dirs.append((int(d), dir_path))
    case_dirs.sort()

    text_count = 0
    visual_count = 0
    skipped = 0

    for case_id, case_path in case_dirs:
        # ── Text ──
        text_file = os.path.join(case_path, "artitle_section.txt")
        text_emb = None
        if os.path.isfile(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                sentences = [s.strip() for s in f.readlines() if s.strip()]
            if sentences:
                text_emb = extract_text_embeddings(sentences, vectorizer)
                np.save(os.path.join(case_path, "text_embeddings.npy"), text_emb)

        # ── Visual ──
        keyframes = sorted(glob.glob(os.path.join(case_path, "*_summary.jpg")))
        visual_emb = None
        if keyframes:
            visual_emb = extract_visual_embeddings(keyframes)
            np.save(os.path.join(case_path, "visual_embeddings.npy"), visual_emb)

        # ── Log ──
        if text_emb is not None or visual_emb is not None:
            t_shape = text_emb.shape if text_emb is not None else "–"
            v_shape = visual_emb.shape if visual_emb is not None else "–"
            if (text_count + visual_count) < 6 or case_id % 50 == 0:
                print(f"  Case {case_id:>3d}  |  text: {str(t_shape):>12s}  |  visual: {str(v_shape):>12s}")
            if text_emb is not None:
                text_count += 1
            if visual_emb is not None:
                visual_count += 1
        else:
            skipped += 1

    # Step 3: Summary
    print(f"\n[3/3] Done!")
    print(f"\n{'=' * 60}")
    print(f"  Text embeddings saved:    {text_count} cases  → text_embeddings.npy")
    print(f"  Visual embeddings saved:  {visual_count} cases → visual_embeddings.npy")
    print(f"  Skipped (no data):        {skipped} cases")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
