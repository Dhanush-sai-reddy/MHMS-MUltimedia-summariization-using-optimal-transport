import os, json, torch, argparse
import numpy as np
from mhms.dataset_unified import CNNMultimodalDatasetUnified
from mhms.models.mhms_framework_unified import MHMS_Unified

def generate_summaries(model, dataset, device, top_k=3):
    model.eval()
    generated = 0
    results_map = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            case_id = dataset.samples[i]["id"]
            
            tf = batch['text_features'].unsqueeze(0).to(device)
            vf = batch['video_features'].unsqueeze(0).to(device)
            tm = batch['text_mask'].unsqueeze(0).to(device)
            vm = batch['video_mask'].unsqueeze(0).to(device)

            matched = model.generate_multimodal_summary_topk(
                text_features=tf, video_features=vf,
                text_mask=tm, video_mask=vm, top_k=top_k
            )[0]
            
            results_map[case_id] = matched
            generated += 1
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{len(dataset)}] case {case_id}")
    return results_map

def evaluate_rouge(results_map, data_dir="cnn_data"):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("rouge-score not installed")
        return None

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for case_id, matched in results_map.items():
        case_path = os.path.join(data_dir, str(case_id))
        article_file = os.path.join(case_path, "artitle_section.txt")
        highlight_file = os.path.join(case_path, "highlight.txt")
        
        if not all(os.path.exists(f) for f in [article_file, highlight_file]):
            continue

        with open(article_file, 'r', encoding='utf-8') as f:
            sentences = [l.strip() for l in f if l.strip()]
        with open(highlight_file, 'r', encoding='utf-8') as f:
            ground_truth = " ".join([l.strip() for l in f if l.strip()])

        idx_list = [m["text_idx"] for m in matched]
        # Keep unique indices and sort them to maintain order
        pred_text = " ".join([sentences[i] for i in sorted(set(idx_list)) if 0 <= i < len(sentences)])
        
        if not pred_text:
            continue

        score = scorer.score(ground_truth, pred_text)
        for k in scores:
            scores[k].append(score[k].fmeasure)

    if not scores['rouge1']:
        return None
    return {k: float(np.mean(v)) for k, v in scores.items()} | {'num_evaluated': len(scores['rouge1'])}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["clip", "qwen2vl", "qwen3vl"], default="clip")
    parser.add_argument("--weights", type=str, default="mhms_clip_final.pth")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--dir", type=str, default="embeddings_clip")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {args.model} on {device}...")

    dataset = CNNMultimodalDatasetUnified(embeddings_dir=args.dir, embedding_dim=args.dim)
    model = MHMS_Unified(embedding_dim=args.dim)
    
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print(f"Warning: {args.weights} not found, using random weights!")
        
    model.to(device)

    print("Generating summaries...")
    results_map = generate_summaries(model, dataset, device)
    
    print("Calculating ROUGE scores...")
    results = evaluate_rouge(results_map)
    
    if results:
        paper = {'rouge1': 28.02, 'rouge2': 8.94, 'rougeL': 18.89}
        print(f"\nResults for {args.model.upper()}:")
        print(f"Evaluated {results['num_evaluated']} cases")
        for k in ['rouge1', 'rouge2', 'rougeL']:
            val = results[k] * 100
            diff = val - paper[k]
            print(f"  {k.upper():>7}: {val:.2f} (Paper: {paper[k]:.2f}, Diff: {diff:+.2f})")

if __name__ == "__main__":
    main()
