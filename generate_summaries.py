import os, json, torch
from mhms.dataset import CNNMultimodalDataset
from mhms.models.mhms_framework import MHMS

def generate_and_save_summaries():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = CNNMultimodalDataset(data_dir="cnn_data", embeddings_dir="embeddings", max_sentences=20, visual_dim=2048, text_dim=768, max_shots=20)
    if len(dataset) == 0:
        return

    model = MHMS(text_feature_dim=768, visual_feature_dim=2048, video_hidden_dim=256, video_omega_b=3)
    if os.path.exists("mhms_model_weights.pth"):
        model.load_state_dict(torch.load("mhms_model_weights.pth", map_location=device))
    model.to(device)
    model.eval()

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
                text_mask=text_mask, video_mask=video_mask, top_k=3
            )[0]

            output = {"multimodal_summary": [
                {"selected_text_sentence_index": m["text_idx"],
                 "aligned_visual_segment_index": m["video_idx"],
                 "optimal_transport_match_mass": round(m["match_score"], 4)}
                for m in matched
            ]}

            with open(os.path.join(sample_path, "multimodal_summary_output.json"), "w") as f:
                json.dump(output, f, indent=4)

            if i % 10 == 0:
                print(f"Generated: {sample_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    generate_and_save_summaries()
