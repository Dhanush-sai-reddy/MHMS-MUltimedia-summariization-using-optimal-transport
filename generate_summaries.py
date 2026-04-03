import os
import torch
import json
from mhms.dataset import CNNMultimodalDataset
from mhms.models.mhms_framework import MHMS

def generate_and_save_summaries():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Dataset
    print("Loading CNN Multimodal Dataset for Inference...")
    dataset = CNNMultimodalDataset(data_dir="cnn_data", max_sentences=20, max_words=64, visual_dim=1024, max_shots=20)
    
    if len(dataset) == 0:
        print("Dataset is empty. Ensure 'cnn_data' has the right folder structure.")
        return

    # 2. Initialize the Model framework
    print("Initializing MHMS Framework...")
    model = MHMS(
        text_hidden_size=256,
        visual_feature_dim=1024,
        video_hidden_dim=256,
        video_omega_b=3 
    )
    
    # If a trained model is available locally, load it. Otherwise, generate summaries randomly using untuned weights.
    if os.path.exists("mhms_model_weights.pth"):
        model.load_state_dict(torch.load("mhms_model_weights.pth", map_location=device))
        print("Pre-trained MHMS weights loaded successfully!")
    else:
        print("No trained weights found. Running generation with untrained initialized weights.")
        
    model.to(device)
    model.eval()

    print("\nStarting generation over cases...\n")
    
    # Process each case item individually
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sample_path = dataset.samples[i]["path"]  # "cnn_data/5"
            
            # Need to add batch dimension (1, ...)
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            video_features = sample['video_features'].unsqueeze(0).to(device)
            
            # Run inference method which leverages OT to fetch alignment pairings
            matched_summaries = model.generate_multimodal_summary(
                input_ids=input_ids,
                attention_mask=attention_mask,
                video_features=video_features,
                threshold=0.45 # Extract sequences with probability score over this percentage length
            )
            
            # Since batch_size is 1
            batch_result = matched_summaries[0]
            
            # Formulate the summary output result
            output_data = {
                "multimodal_summary": []
            }
            
            # Save the highest scored Optimal Transport pairs
            for match in batch_result:
                output_data["multimodal_summary"].append({
                    "selected_text_sentence_index": match["text_idx"],
                    "aligned_visual_segment_index": match["video_idx"],
                    "optimal_transport_match_mass": round(match["match_score"], 4)
                })
                
            # Write JSON output directly inside the sample's case folder
            save_dest = os.path.join(sample_path, "multimodal_summary_output.json")
            with open(save_dest, "w") as f:
                json.dump(output_data, f, indent=4)
                
            if i % 10 == 0:
                print(f"Successfully processed and generated summary for target: {sample_path}")

    print("\nFinished generating and saving multimodal summaries inside each cnn_data folder!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    generate_and_save_summaries()
