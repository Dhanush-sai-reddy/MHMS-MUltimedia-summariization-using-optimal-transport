import os
import json
import glob

def build_report():
    data_dir = "cnn_data"
    output_html = "multimodal_showcase.html"
    
    html_content = """
    <html>
    <head>
        <title>MHMS Multimodal Summaries Showcase</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f7f9; color: #333; }
            h1 { text-align: center; color: #2c3e50; }
            .case-container { background: #fff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; padding: 20px; overflow: hidden;}
            .case-title { font-size: 1.2em; font-weight: bold; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px;}
            .content-wrapper { display: flex; gap: 30px; }
            .text-section { flex: 1; font-size: 1.1em; line-height: 1.6; }
            .visual-section { flex: 1; text-align: center; }
            img { max-width: 100%; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
            .ground-truth { margin-top: 20px; padding: 15px; background: #e8f4f8; border-left: 4px solid #3498db; font-style: italic; font-size: 0.95em;}
        </style>
    </head>
    <body>
        <h1>MHMS Multimodal Summaries Showcase</h1>
    """
    
    count = 0
    for case_folder in sorted(os.listdir(data_dir)):
        case_path = os.path.join(data_dir, case_folder)
        if not os.path.isdir(case_path):
            continue
            
        json_file = os.path.join(case_path, "multimodal_summary_output.json")
        article_file = os.path.join(case_path, "artitle_section.txt")
        highlight_file = os.path.join(case_path, "highlight.txt")
        
        if not (os.path.exists(json_file) and os.path.exists(article_file)):
            continue
            
        with open(json_file, 'r') as f:
            predictions = json.load(f)
            
        with open(article_file, 'r', encoding='utf-8', errors='ignore') as f:
            sentences = [line.strip() for line in f.readlines() if line.strip()]
            
        gt_text = ""
        if os.path.exists(highlight_file):
            with open(highlight_file, 'r', encoding='utf-8', errors='ignore') as f:
                gt_text = " ".join([line.strip() for line in f.readlines() if line.strip()])
                
        # Get generated summary
        idx_list = [item["selected_text_sentence_index"] for item in predictions.get("multimodal_summary", [])]
        selected_sents = [sentences[idx] for idx in sorted(set(idx_list)) if 0 <= idx < len(sentences)]
        pred_text = " ".join(selected_sents)
        
        if not pred_text:
            continue
            
        # Find exactly one image from the same case folder to pair with it
        video_idx_list = [item["aligned_visual_segment_index"] for item in predictions.get("multimodal_summary", [])]
        image_path = ""
        
        # We try to find the specific segment index, but fall back to any summary image
        images = glob.glob(os.path.join(case_path, "*_summary.jpg"))
        if images:
            image_path = images[0]
            # Try to match video index exactly if possible
            for v_idx in video_idx_list:
                potential_match = os.path.join(case_path, f"segment{v_idx}_summary.jpg")
                if os.path.exists(potential_match):
                    image_path = potential_match
                    break
        
        # Format HTML relative path properly for the browser
        img_tag = f'<img src="{os.path.relpath(image_path, ".").replace(chr(92), "/")}" alt="Cover Image">' if image_path else '<div style="padding: 50px; background: #eee; border-radius: 5px;">No visual keyframe extracted</div>'
        
        html_content += f"""
        <div class="case-container">
            <div class="case-title">Case Document {case_folder}</div>
            <div class="content-wrapper">
                <div class="text-section">
                    <strong>Generated Cover Text:</strong><br><br>
                    {pred_text}
                    <div class="ground-truth">
                        <strong>Ground Truth Highlights:</strong><br>
                        {gt_text}
                    </div>
                </div>
                <div class="visual-section">
                    <strong>Assigned Cover Image:</strong><br><br>
                    {img_tag}
                </div>
            </div>
        </div>
        """
        count += 1
        if count >= 30: # Limit to 30 to keep HTML reasonable sizes
            break
            
    html_content += """
    </body>
    </html>
    """
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Constructed Multi-Modal Showcase HTML Report at {output_html} with {count} examples.")

if __name__ == "__main__":
    build_report()
