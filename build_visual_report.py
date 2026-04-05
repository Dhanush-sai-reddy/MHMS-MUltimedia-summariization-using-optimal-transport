import os
import json
import glob
import textwrap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure


def build_report():
    data_dir = "cnn_data"
    output_img = "multimodal_showcase.png"

    cases_to_plot = []

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

        # Find image
        video_idx_list = [item["aligned_visual_segment_index"] for item in predictions.get("multimodal_summary", [])]
        image_path = ""

        images = glob.glob(os.path.join(case_path, "*_summary.jpg"))
        if images:
            image_path = images[0]
            for v_idx in video_idx_list:
                potential_match = os.path.join(case_path, f"segment{v_idx}_summary.jpg")
                if os.path.exists(potential_match):
                    image_path = potential_match
                    break

        cases_to_plot.append((case_folder, pred_text, gt_text, image_path))
        if len(cases_to_plot) >= 4:
            break

    if not cases_to_plot:
        print("No cases found to visualize.")
        return

    # =====================================================
    # APPROACH: One separate figure per case, saved individually,
    # then optionally combined. This guarantees ZERO overlap.
    # =====================================================

    all_figs = []

    for i, (case_folder, pred_text, gt_text, image_path) in enumerate(cases_to_plot):

        # --- Prepare text ---
        wrap_w = 75
        wrapped_pred = textwrap.fill(pred_text, width=wrap_w)
        wrapped_gt = textwrap.fill(gt_text, width=wrap_w)

        text_block = (
            f"GENERATED COVER TEXT:\n"
            f"{'─' * 40}\n"
            f"{wrapped_pred}\n\n"
            f"GROUND TRUTH HIGHLIGHTS:\n"
            f"{'─' * 40}\n"
            f"{wrapped_gt}"
        )

        # Count lines to determine figure height
        num_lines = text_block.count('\n') + 1
        # Each line needs ~0.22 inches at fontsize 10, plus padding
        text_height_inches = max(num_lines * 0.22, 4.0)
        fig_height = max(text_height_inches + 1.5, 5.0)  # extra for title + padding

        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(18, fig_height))

        # --- Left: Image ---
        ax_img.axis('off')
        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            ax_img.imshow(img)
        else:
            ax_img.text(0.5, 0.5, "No Image Available", fontsize=14,
                        ha='center', va='center', style='italic', color='gray')
            ax_img.set_xlim(0, 1)
            ax_img.set_ylim(0, 1)
        ax_img.set_title(f"Document: {case_folder}  (Keyframe)",
                         fontsize=16, fontweight='bold', pad=12)

        # --- Right: Text ---
        ax_text.axis('off')
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        ax_text.text(0.02, 0.98, text_block,
                     transform=ax_text.transAxes,
                     fontsize=10, family='monospace', linespacing=1.4,
                     verticalalignment='top', horizontalalignment='left')
        ax_text.set_title("Summary Text", fontsize=14, fontweight='bold', pad=12)

        fig.tight_layout(pad=2.0)

        # Save each case as its own file
        case_file = f"multimodal_case_{i}.png"
        fig.savefig(case_file, dpi=150, bbox_inches='tight', facecolor='white')
        all_figs.append((fig, case_file))
        print(f"  Saved {case_file}")

    # --- Stitch all case images into one tall combined image ---
    from PIL import Image

    case_images = []
    for _, case_file in all_figs:
        case_images.append(Image.open(case_file))

    total_width = max(img.width for img in case_images)
    # Add a 30px gap between each case
    gap = 30
    total_height = sum(img.height for img in case_images) + gap * (len(case_images) - 1)

    combined = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for img in case_images:
        # Center horizontally if widths differ
        x_offset = (total_width - img.width) // 2
        combined.paste(img, (x_offset, y_offset))
        y_offset += img.height + gap

    combined.save(output_img)
    print(f"\nConstructed Multi-Modal Showcase at {output_img} with {len(cases_to_plot)} examples.")

    # Show the combined result
    plt.figure(figsize=(16, 20))
    plt.axis('off')
    plt.imshow(combined)
    plt.tight_layout()
    plt.show()

    # Clean up individual case files
    for _, case_file in all_figs:
        os.remove(case_file)


if __name__ == "__main__":
    build_report()
