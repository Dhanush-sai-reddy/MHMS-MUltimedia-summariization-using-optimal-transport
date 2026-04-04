# MHMS — Multimodal Hierarchical Multimedia Summarization using Optimal Transport

Implementation of the MHMS framework from [arXiv:2204.03734](https://arxiv.org/abs/2204.03734) by Qiu et al.

> Multimedia summarization with multimodal output — automatically generating cover images and titles for news articles by aligning visual and textual domains through **Optimal Transport (Sinkhorn-Knopp algorithm)**.

---

## Architecture

```
CNN Dataset (257 articles + 205 videos)
        │
        ├── Text Pipeline
        │     artitle_section.txt → TF-IDF → text_embeddings.npy
        │     highlight.txt (ground truth summaries)
        │     label.txt (extractive labels)
        │
        ├── Video Pipeline
        │     video/*.ts → K-Means + Laplacian → *_summary.jpg (keyframes)
        │     Keyframe images → HSV histogram + texture → visual_embeddings.npy
        │
        └── Optimal Transport Alignment
              text_embeddings ←→ visual_embeddings
              Cosine distance matrix → Sinkhorn algorithm → Transport plan T
              T[i,j] = alignment strength between sentence i and keyframe j
```

---

## Project Files

### Core Pipeline

- [embedding_pipeline.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/embedding_pipeline.py) — Extracts TF-IDF text embeddings and visual feature embeddings from the CNN dataset, stores centrally in `embeddings/`.
- [optimal_transport.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/optimal_transport.py) — Sinkhorn-Knopp algorithm for solving the entropic regularized Optimal Transport problem between modalities.
- [generate_video_summaries.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/generate_video_summaries.py) — Extracts keyframes from `.ts` video segments using K-Means clustering + Laplacian sharpness selection.
- [generate_summaries.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/generate_summaries.py) — Generates OT-aligned multimodal summary pairs and saves `multimodal_summary_output.json` per case.
- [train.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/train.py) — Trains the full MHMS neural framework with BCE text loss + OT cross-modal alignment loss.

### Framework Modules

- [mhms/dataset.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/mhms/dataset.py) — CNN multimodal dataset loader with text tokenization and video feature handling.
- [mhms/models/mhms_framework.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/mhms/models/mhms_framework.py) — Full MHMS model: BiGRU text encoder, VTS video segmenter, summarizers, and differentiable Sinkhorn OT alignment.
- [mhms/models/text_segmentation.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/mhms/models/text_segmentation.py) — HierarchicalBERT: sentence-level BERT + article-level Transformer for text segmentation boundaries.
- [mhms/models/video_temporal_segmentation.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/mhms/models/video_temporal_segmentation.py) — VTS module with VTS_d (difference) and VTS_r (relation) branches + Bi-LSTM boundary scoring.
- [mhms/models/summarization.py](https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport/blob/main/mhms/models/summarization.py) — Text extractive summarizer (BiLSTM) and Visual encoder-decoder summarizer with attention.

---

## Setup

```bash
git clone https://github.com/Dhanush-sai-reddy/MHMS-MUltimedia-summariization-using-optimal-transport.git
cd MHMS-MUltimedia-summariization-using-optimal-transport
pip install -r requirements.txt
```

**Dependencies**: `torch`, `numpy`, `opencv-python`, `scikit-learn`, `transformers`

## Usage

### 1. Extract Keyframes from Videos

```bash
python generate_video_summaries.py
```

Processes `video/*.ts` segments → K-Means clustering by color histogram → selects sharpest frame per cluster → saves `*_summary.jpg`.

### 2. Generate Embeddings

```bash
python embedding_pipeline.py
```

Extracts text (TF-IDF, 512-dim) and visual (HSV histogram + texture, 576-dim) embeddings. Outputs stored in `embeddings/text/` and `embeddings/visual/`.

### 3. Run Optimal Transport Alignment

```bash
python optimal_transport.py
```

Demonstrates the Sinkhorn-Knopp algorithm: cosine distance cost matrix → entropic regularization → optimal transport plan `T`.

### 4. Generate Multimodal Summaries

```bash
python generate_summaries.py
```

Uses the MHMS framework to produce OT-aligned text↔visual summary pairs as JSON.

---

## Key Algorithms

### Optimal Transport (Sinkhorn-Knopp)

1. **Cost Matrix**: `C[k,m] = 1 - cos(e_k, v_m)` — cosine distance between text embedding `e_k` and visual embedding `v_m`
2. **Gibbs Kernel**: `K = exp(-C / λ)` — entropic regularization
3. **Sinkhorn Iterations**: Alternating row/column normalization until convergence
4. **Transport Plan**: `T = diag(u) · K · diag(v)` — high `T[i,j]` = strong text↔visual alignment

### Video Temporal Segmentation (VTS)

- **VTS_d**: Temporal convolutions on before/after windows → inner product for boundary detection
- **VTS_r**: Temporal convolution + max pooling over full window for context
- **Bi-LSTM**: Sequence modeling over boundary candidates → probability scores

---

## Data Format

```
cnn_data/
├── label.txt                    # Extractive summary labels (444 entries)
└── <case_id>/                   # Numbered 1-262
    ├── artitle.txt              # Raw article text
    ├── artitle_section.txt      # Article split into sentences
    ├── highlight.txt            # Ground truth highlights
    ├── title.txt                # Article title
    ├── transcript.txt           # Video transcript (if available)
    ├── video/*.ts               # Video segments
    └── *_summary.jpg            # Extracted keyframes
```

---

## Paper Reference

```bibtex
@article{qiu2022mhms,
  title={MHMS: Multimodal Hierarchical Multimedia Summarization},
  author={Qiu, Jielin and Zhu, Jiacheng and Xu, Mengdi and Dernoncourt, Franck 
          and Bui, Trung and Wang, Zhaowen and Li, Bo and Zhao, Ding and Jin, Hailin},
  journal={arXiv preprint arXiv:2204.03734},
  year={2022}
}
```