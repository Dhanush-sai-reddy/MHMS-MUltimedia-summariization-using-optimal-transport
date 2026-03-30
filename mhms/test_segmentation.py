import torch
import os
from models.text_segmentation import HierarchicalBERT
from models.video_temporal_segmentation import VTS

def run_tests_and_save():
    os.makedirs("outputs", exist_ok=True)
    
    print("=== Testing Textual Segmentation (HierarchicalBERT) ===")
    
    try:
        # Initialize text segmentation model. 
        # Using fewer transformer layers (2) just to instantiate quickly for the test.
        text_model = HierarchicalBERT(pretrained_model_name='bert-base-uncased', num_article_layers=2)
        text_model.eval()
        
        # Create a dummy batch: 1 Article, 5 Sentences, 12 tokens per sentence
        dummy_input_ids = torch.randint(0, 30522, (1, 5, 12))  
        dummy_attention_mask = torch.ones(1, 5, 12)
        
        with torch.no_grad():
            text_probs, article_features = text_model(dummy_input_ids, dummy_attention_mask)
            
        print(f"Successfully processed text!\n -> Boundary Probs Shape: {text_probs.shape}\n -> Article Features Shape: {article_features.shape}")
        
        torch.save({
            "segmentation_probabilities": text_probs,
            "article_transformer_features": article_features
        }, "outputs/text_segmentation_tensors.pt")
        print("-> Saved pure text tensors to 'outputs/text_segmentation_tensors.pt'\n")

    except Exception as e:
        print(f"Textual Model Test Failed: {e}\n")


    print("=== Testing Video Temporal Segmentation (VTS) ===")
    try:
        # Initialize video segmentation model with arbitrary feature sizes
        video_model = VTS(visual_feature_dim=1024, hidden_dim=256, omega_b=3)
        video_model.eval()
        
        # Create dummy batch: 1 Video, 20 shots/frames, 1024 feature dimension
        dummy_video_features = torch.randn(1, 20, 1024)
        
        with torch.no_grad():
            video_probs = video_model(dummy_video_features)
            
        print(f"Successfully processed video!\n -> Video Boundary Probs Shape: {video_probs.shape} (Expect Seq Length - 2*omega_b)")
        
        torch.save({
            "video_boundary_probabilities": video_probs
        }, "outputs/video_segmentation_tensors.pt")
        print("-> Saved pure video tensors to 'outputs/video_segmentation_tensors.pt'\n")
        
    except Exception as e:
        print(f"Video Model Test Failed: {e}\n")

if __name__ == "__main__":
    run_tests_and_save()
