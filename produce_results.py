import os, json, torch
import numpy as np
from mhms.dataset import CNNMultimodalDataset
from mhms.models.mhms_framework import MHMS


def generate_all_summaries(model, dataset, device, top_k=3):
    model.eval()
    generated = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            case_id = dataset.samples[i]["id"]
            sample_path = os.path.join("cnn_data", str(case_id))

            text_features = sample['text_features'].unsqueeze(0).to(device)
            video_features = sample['video_features'].unsqueeze(0).to(device)
            text_mask = sample['text_mask'].unsqueeze(0).to(device)
            video_mask = sample['video_mask'].unsqueeze(0).to(device)

            matched = model.generate_multimodal_summary_topk(
                text_features=text_features, video_features=video_features,
                text_mask=text_mask, video_mask=video_mask, top_k=top_k
            )[0]

            output = {"multimodal_summary": [
                {"selected_text_sentence_index": m["text_idx"],
                 "aligned_visual_segment_index": m["video_idx"],
                 "optimal_transport_match_mass": round(m["match_score"], 6),
                 "text_summarizer_prob": round(m["text_prob"], 4),
                 "video_summarizer_prob": round(m["video_prob"], 4)}
                for m in matched
            ]}
            with open(os.path.join(sample_path, "multimodal_summary_output.json"), "w") as f:
                json.dump(output, f, indent=4)
            generated += 1
            if i % 50 == 0:
                print(f"  [{i+1}/{len(dataset)}] case {case_id}")
    return generated


def evaluate_rouge(data_dir="cnn_data"):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("rouge-score not installed")
        return None

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for case_folder in sorted(os.listdir(data_dir), key=lambda x: int(x) if x.isdigit() else 0):
        case_path = os.path.join(data_dir, case_folder)
        if not os.path.isdir(case_path):
            continue
        json_file = os.path.join(case_path, "multimodal_summary_output.json")
        article_file = os.path.join(case_path, "artitle_section.txt")
        highlight_file = os.path.join(case_path, "highlight.txt")
        if not all(os.path.exists(f) for f in [json_file, article_file, highlight_file]):
            continue

        with open(json_file) as f:
            predictions = json.load(f)
        with open(article_file) as f:
            sentences = [l.strip() for l in f if l.strip()]
        with open(highlight_file) as f:
            ground_truth = " ".join([l.strip() for l in f if l.strip()])

        idx_list = [item["selected_text_sentence_index"] for item in predictions.get("multimodal_summary", [])]
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
    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MHMS(text_feature_dim=768, visual_feature_dim=2048, video_hidden_dim=256, video_omega_b=3)
    if os.path.exists("mhms_model_weights.pth"):
        model.load_state_dict(torch.load("mhms_model_weights.pth", map_location=device))
    model.to(device)

    dataset = CNNMultimodalDataset(data_dir="cnn_data", embeddings_dir="embeddings",
                                   max_sentences=20, visual_dim=2048, text_dim=768, max_shots=20)
    generated = generate_all_summaries(model, dataset, device, top_k=3)
    print(f"Generated {generated} summaries.")

    results = evaluate_rouge()
    if results:
        paper = {'rouge1': 28.02, 'rouge2': 8.94, 'rougeL': 18.89}
        print(f"\nEvaluated {results['num_evaluated']} cases:")
        for k in ['rouge1', 'rouge2', 'rougeL']:
            ours = results[k] * 100
            print(f"  {k.upper()}: {ours:.2f} (paper: {paper[k]:.2f})")

        os.makedirs("results", exist_ok=True)
        with open("results/evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
