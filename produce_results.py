"""
MHMS — Full Results Pipeline
============================
Generates multimodal summaries using trained weights, then evaluates 
against ground truth highlights using ROUGE metrics.

Usage:
    python produce_results.py
"""
import os
import sys
import json
import torch
import numpy as np
from mhms.dataset import CNNMultimodalDataset
from mhms.models.mhms_framework import MHMS


def generate_all_summaries(model, dataset, device, top_k=3):
    """Generate multimodal summaries for all cases in the dataset."""
    model.eval()
    generated = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            case_id = dataset.samples[i]["id"]
            sample_path = os.path.join("cnn_data", str(case_id))
            
            text_features = sample['text_features'].unsqueeze(0).to(device)
            video_features = sample['video_features'].unsqueeze(0).to(device)
            
            matched_summaries = model.generate_multimodal_summary_topk(
                text_features=text_features,
                video_features=video_features,
                top_k=top_k
            )
            
            batch_result = matched_summaries[0]
            
            output_data = {"multimodal_summary": []}
            for match in batch_result:
                output_data["multimodal_summary"].append({
                    "selected_text_sentence_index": match["text_idx"],
                    "aligned_visual_segment_index": match["video_idx"],
                    "optimal_transport_match_mass": round(match["match_score"], 6),
                    "text_summarizer_prob": round(match["text_prob"], 4),
                    "video_summarizer_prob": round(match["video_prob"], 4)
                })
            
            save_dest = os.path.join(sample_path, "multimodal_summary_output.json")
            with open(save_dest, "w") as f:
                json.dump(output_data, f, indent=4)
            
            generated += 1
            if i % 50 == 0:
                print(f"  [{i+1}/{len(dataset)}] Generated summary for case {case_id}")
    
    return generated


def evaluate_rouge(data_dir="cnn_data"):
    """Evaluate generated summaries against ground truth using ROUGE."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("\n  [!] rouge-score not installed. Install with: pip install rouge-score")
        return None
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    evaluated = 0
    for case_folder in sorted(os.listdir(data_dir), key=lambda x: int(x) if x.isdigit() else 0):
        case_path = os.path.join(data_dir, case_folder)
        if not os.path.isdir(case_path):
            continue
        
        json_file = os.path.join(case_path, "multimodal_summary_output.json")
        article_file = os.path.join(case_path, "artitle_section.txt")
        highlight_file = os.path.join(case_path, "highlight.txt")
        
        if not all(os.path.exists(f) for f in [json_file, article_file, highlight_file]):
            continue
        
        with open(json_file, 'r') as f:
            predictions = json.load(f)
        
        with open(article_file, 'r') as f:
            sentences = [line.strip() for line in f.readlines() if line.strip()]
        
        with open(highlight_file, 'r') as f:
            ground_truth = " ".join([line.strip() for line in f.readlines() if line.strip()])
        
        # Extract unique selected sentence indices, sorted
        idx_list = [item["selected_text_sentence_index"] for item in predictions.get("multimodal_summary", [])]
        selected_sents = []
        for idx in sorted(set(idx_list)):
            if 0 <= idx < len(sentences):
                selected_sents.append(sentences[idx])
        
        pred_text = " ".join(selected_sents)
        if not pred_text:
            continue
        
        score = scorer.score(ground_truth, pred_text)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
        evaluated += 1
    
    if evaluated == 0:
        return None
    
    return {
        'num_evaluated': evaluated,
        'rouge1': float(np.mean(scores['rouge1'])),
        'rouge2': float(np.mean(scores['rouge2'])),
        'rougeL': float(np.mean(scores['rougeL'])),
        'rouge1_std': float(np.std(scores['rouge1'])),
        'rouge2_std': float(np.std(scores['rouge2'])),
        'rougeL_std': float(np.std(scores['rougeL']))
    }


def print_results(results, generated_count):
    """Print a formatted results table."""
    print("\n" + "=" * 70)
    print("  MHMS — RESULTS REPORT")
    print("=" * 70)
    
    print(f"\n  Pipeline Summary:")
    print(f"    Summaries generated:  {generated_count}")
    print(f"    Cases evaluated:      {results['num_evaluated']}")
    
    print(f"\n  {'─' * 60}")
    print(f"  {'Metric':<15} {'Our Result':>12} {'Paper (CNN)':>14} {'Gap':>10}")
    print(f"  {'─' * 60}")
    
    paper = {'rouge1': 28.02, 'rouge2': 8.94, 'rougeL': 18.89}
    
    for metric, paper_val in paper.items():
        our_val = results[metric] * 100  # Convert to percentage
        our_std = results[f'{metric}_std'] * 100
        gap = our_val - paper_val
        label = metric.upper().replace('ROUGE', 'ROUGE-')
        print(f"  {label:<15} {our_val:>8.2f} ±{our_std:>4.2f} {paper_val:>12.2f} {gap:>+10.2f}")
    
    print(f"  {'─' * 60}")
    
    print(f"\n  Notes:")
    print(f"    - Paper uses BART-Large-CNN (abstractive), we use BiLSTM (extractive)")
    print(f"    - Paper trained 10-20 epochs on 4×V100, we trained 3 epochs on CPU")
    print(f"    - Paper uses batch_size=32, we use batch_size=2")
    print(f"    - Lower scores expected due to above differences")
    print("=" * 70)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load model
    print("\n[1/3] Loading MHMS model...")
    model = MHMS(
        text_feature_dim=768,
        visual_feature_dim=2048,
        video_hidden_dim=256,
        video_omega_b=3
    )
    
    weights_path = "mhms_model_weights.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"  Loaded trained weights from {weights_path}")
    else:
        print(f"  [!] No trained weights found. Using random initialization.")
    
    model.to(device)
    
    # 2. Load dataset and generate summaries
    print("\n[2/3] Generating multimodal summaries (Top-K=3)...")
    dataset = CNNMultimodalDataset(
        data_dir="cnn_data", embeddings_dir="embeddings",
        max_sentences=20, visual_dim=2048, text_dim=768, max_shots=20
    )
    
    generated = generate_all_summaries(model, dataset, device, top_k=3)
    print(f"  Generated {generated} multimodal summaries.")
    
    # 3. Evaluate with ROUGE
    print("\n[3/3] Evaluating with ROUGE metrics...")
    results = evaluate_rouge()
    
    if results:
        print_results(results, generated)
        
        # Save results to file
        os.makedirs("results", exist_ok=True)
        results_path = os.path.join("results", "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {results_path}")
    else:
        print("  [!] No cases could be evaluated. Check data availability.")


if __name__ == "__main__":
    main()
