import os
import json
from rouge_score import rouge_scorer
import numpy as np

def evaluate():
    data_dir = "cnn_data"
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for case_folder in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case_folder)
        if not os.path.isdir(case_path):
            continue
            
        json_file = os.path.join(case_path, "multimodal_summary_output.json")
        article_file = os.path.join(case_path, "artitle_section.txt")
        highlight_file = os.path.join(case_path, "highlight.txt")
        
        if not (os.path.exists(json_file) and os.path.exists(article_file) and os.path.exists(highlight_file)):
            continue
            
        with open(json_file, 'r') as f:
            predictions = json.load(f)
            
        with open(article_file, 'r') as f:
            sentences = [line.strip() for line in f.readlines() if line.strip()]
            
        with open(highlight_file, 'r') as f:
            ground_truth = " ".join([line.strip() for line in f.readlines() if line.strip()])
            
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
        
    if len(scores['rouge1']) > 0:
        print(f"Evaluated {len(scores['rouge1'])} cases.")
        print(f"ROUGE-1 F-Measure: {np.mean(scores['rouge1']):.4f}")
        print(f"ROUGE-2 F-Measure: {np.mean(scores['rouge2']):.4f}")
        print(f"ROUGE-L F-Measure: {np.mean(scores['rougeL']):.4f}")
    else:
        print("No paired summaries found to evaluate.")

if __name__ == "__main__":
    evaluate()
