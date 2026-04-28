"""
Build a rich, self-contained multimodal_showcase.html with ALL evaluated cases,
ROUGE scores, OT alignment metrics, per-case keyframes, and pagination.
"""
import os, json, glob, base64, io
from pathlib import Path

DATA_DIR = "cnn_data"
RESULTS_FILE = "results/evaluation_results.json"
OUTPUT_HTML = "multimodal_showcase.html"
MAX_CASES = 204  # show all evaluated cases


def img_to_base64(path, max_w=480):
    """Read image, resize, return base64 data-URI."""
    try:
        from PIL import Image
        img = Image.open(path)
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def load_cases():
    cases = []
    for d in sorted(os.listdir(DATA_DIR), key=lambda x: int(x) if x.isdigit() else 0):
        cp = os.path.join(DATA_DIR, d)
        if not os.path.isdir(cp):
            continue
        jf = os.path.join(cp, "multimodal_summary_output.json")
        af = os.path.join(cp, "artitle_section.txt")
        hf = os.path.join(cp, "highlight.txt")
        if not (os.path.exists(jf) and os.path.exists(af) and os.path.exists(hf)):
            continue

        with open(jf) as f:
            pred = json.load(f)
        with open(af, encoding="utf-8", errors="ignore") as f:
            sents = [l.strip() for l in f if l.strip()]
        with open(hf, encoding="utf-8", errors="ignore") as f:
            gt = " ".join([l.strip() for l in f if l.strip()])

        summary_items = pred.get("multimodal_summary", [])
        idx_list = [item["selected_text_sentence_index"] for item in summary_items]
        sel = [sents[i] for i in sorted(set(idx_list)) if 0 <= i < len(sents)]
        pred_text = " ".join(sel)
        if not pred_text:
            continue

        # Title
        title = ""
        tf = os.path.join(cp, "title.txt")
        if os.path.exists(tf):
            with open(tf, encoding="utf-8", errors="ignore") as f:
                title = f.read().strip()

        # Best keyframe image
        video_idx_list = [item["aligned_visual_segment_index"] for item in summary_items]
        image_b64 = ""
        images = glob.glob(os.path.join(cp, "*_summary.jpg"))
        if images:
            chosen = images[0]
            for v_idx in video_idx_list:
                pot = os.path.join(cp, f"segment{v_idx}_summary.jpg")
                if os.path.exists(pot):
                    chosen = pot
                    break
            image_b64 = img_to_base64(chosen)

        # OT stats
        top_ot = max([item["optimal_transport_match_mass"] for item in summary_items], default=0)
        avg_text_prob = sum(item["text_summarizer_prob"] for item in summary_items) / max(len(summary_items), 1)
        avg_vid_prob = sum(item["video_summarizer_prob"] for item in summary_items) / max(len(summary_items), 1)

        # Per-case ROUGE (compute inline)
        rouge_scores = None
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            sc = scorer.score(gt, pred_text)
            rouge_scores = {k: round(sc[k].fmeasure * 100, 2) for k in ["rouge1", "rouge2", "rougeL"]}
        except Exception:
            pass

        cases.append({
            "id": d, "title": title, "pred": pred_text, "gt": gt,
            "img_b64": image_b64, "top_ot": top_ot,
            "avg_text_prob": avg_text_prob, "avg_vid_prob": avg_vid_prob,
            "rouge": rouge_scores, "n_images": len(images),
        })
        if len(cases) >= MAX_CASES:
            break
    return cases


def build_html(cases):
    # Load aggregate ROUGE
    agg = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            agg = json.load(f)

    paper = {"rouge1": 28.02, "rouge2": 8.94, "rougeL": 18.89}

    # Build case cards JSON for JS
    cases_json = json.dumps([{
        "id": c["id"], "title": c["title"],
        "pred": c["pred"], "gt": c["gt"],
        "img": c["img_b64"],
        "ot": round(c["top_ot"], 6),
        "tp": round(c["avg_text_prob"], 4),
        "vp": round(c["avg_vid_prob"], 4),
        "r1": c["rouge"]["rouge1"] if c["rouge"] else None,
        "r2": c["rouge"]["rouge2"] if c["rouge"] else None,
        "rL": c["rouge"]["rougeL"] if c["rouge"] else None,
        "ni": c["n_images"],
    } for c in cases], ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MHMS Multimodal Showcase — Full Results</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --bg:#0f1117;--card:#181b24;--card-hover:#1e2230;
  --accent:#6c63ff;--accent2:#00d4aa;--accent3:#ff6b6b;
  --text:#e4e6ef;--text2:#8b8fa3;--border:#2a2d3a;
  --glow:rgba(108,99,255,0.15);
}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh}}

/* === HEADER === */
.hero{{background:linear-gradient(135deg,#0f1117 0%,#1a1040 50%,#0f1117 100%);padding:60px 40px 40px;text-align:center;border-bottom:1px solid var(--border);position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(circle at 50% 50%,rgba(108,99,255,0.08) 0%,transparent 50%);animation:pulse 8s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.1)}}}}
.hero h1{{font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#6c63ff,#00d4aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px;position:relative}}
.hero .sub{{color:var(--text2);font-size:1rem;font-weight:400;position:relative}}

/* === STATS BAR === */
.stats-bar{{display:flex;justify-content:center;gap:24px;flex-wrap:wrap;padding:28px 40px;background:var(--card);border-bottom:1px solid var(--border)}}
.stat-card{{background:linear-gradient(135deg,rgba(108,99,255,0.1),rgba(0,212,170,0.05));border:1px solid var(--border);border-radius:12px;padding:16px 28px;text-align:center;min-width:160px}}
.stat-card .label{{font-size:0.75rem;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}}
.stat-card .value{{font-size:1.6rem;font-weight:700;color:var(--accent)}}
.stat-card .value.green{{color:var(--accent2)}}
.stat-card .compare{{font-size:0.7rem;color:var(--text2);margin-top:2px}}

/* === CONTROLS === */
.controls{{display:flex;align-items:center;gap:16px;padding:20px 40px;flex-wrap:wrap}}
.search-box{{flex:1;min-width:200px;padding:10px 16px;background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:0.9rem;outline:none;transition:border-color 0.2s}}
.search-box:focus{{border-color:var(--accent)}}
.page-info{{color:var(--text2);font-size:0.85rem;white-space:nowrap}}
.btn{{padding:8px 18px;border:1px solid var(--border);border-radius:8px;background:var(--card);color:var(--text);cursor:pointer;font-size:0.85rem;transition:all 0.2s}}
.btn:hover{{background:var(--accent);border-color:var(--accent);color:#fff}}
.btn:disabled{{opacity:0.3;cursor:default}}
.btn:disabled:hover{{background:var(--card);color:var(--text);border-color:var(--border)}}
.sort-select{{padding:10px 14px;background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:0.85rem;outline:none}}

/* === CASE GRID === */
.grid{{display:grid;grid-template-columns:1fr;gap:20px;padding:0 40px 40px}}
@media(min-width:1200px){{.grid{{grid-template-columns:1fr 1fr}}}}

.case-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden;transition:all 0.3s ease;display:flex;flex-direction:column}}
.case-card:hover{{border-color:var(--accent);box-shadow:0 0 30px var(--glow);transform:translateY(-2px)}}

.case-header{{display:flex;align-items:center;gap:12px;padding:16px 20px;border-bottom:1px solid var(--border);background:rgba(108,99,255,0.04)}}
.case-id{{font-weight:700;font-size:0.85rem;color:var(--accent);background:rgba(108,99,255,0.12);padding:4px 10px;border-radius:6px;white-space:nowrap}}
.case-title{{font-weight:600;font-size:0.95rem;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.ot-badge{{font-size:0.7rem;color:var(--accent2);background:rgba(0,212,170,0.1);padding:3px 8px;border-radius:5px;white-space:nowrap}}

.case-body{{display:flex;flex:1;gap:0}}
.img-col{{width:200px;min-width:200px;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden}}
.img-col img{{width:100%;height:100%;object-fit:cover}}
.img-col .no-img{{color:var(--text2);font-size:0.8rem;font-style:italic;padding:20px;text-align:center}}
.text-col{{flex:1;padding:16px 20px;display:flex;flex-direction:column;gap:12px;overflow:hidden}}

.section-label{{font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;font-weight:600}}
.section-label.gen{{color:var(--accent)}}
.section-label.gt{{color:var(--accent2)}}
.section-text{{font-size:0.82rem;color:var(--text);line-height:1.55;max-height:90px;overflow:hidden;position:relative}}
.section-text::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:30px;background:linear-gradient(transparent,var(--card))}}

.case-metrics{{display:flex;gap:8px;padding:10px 20px;border-top:1px solid var(--border);flex-wrap:wrap}}
.metric{{font-size:0.7rem;padding:3px 8px;border-radius:5px;background:rgba(255,255,255,0.04);color:var(--text2)}}
.metric b{{color:var(--text);font-weight:600}}

/* === FOOTER === */
.footer{{text-align:center;padding:30px;color:var(--text2);font-size:0.8rem;border-top:1px solid var(--border)}}
</style>
</head>
<body>

<div class="hero">
  <h1>MHMS Multimodal Showcase</h1>
  <p class="sub">Multimodal Hierarchical Multimedia Summarization via Optimal Transport &mdash; {len(cases)} evaluated cases</p>
</div>

<div class="stats-bar" id="statsBar">
  <div class="stat-card">
    <div class="label">Cases Evaluated</div>
    <div class="value">{agg.get('num_evaluated', len(cases))}</div>
  </div>
  <div class="stat-card">
    <div class="label">ROUGE-1</div>
    <div class="value green">{agg.get('rouge1',0)*100:.2f}</div>
    <div class="compare">Paper: {paper['rouge1']}</div>
  </div>
  <div class="stat-card">
    <div class="label">ROUGE-2</div>
    <div class="value green">{agg.get('rouge2',0)*100:.2f}</div>
    <div class="compare">Paper: {paper['rouge2']}</div>
  </div>
  <div class="stat-card">
    <div class="label">ROUGE-L</div>
    <div class="value green">{agg.get('rougeL',0)*100:.2f}</div>
    <div class="compare">Paper: {paper['rougeL']}</div>
  </div>
</div>

<div class="controls">
  <input class="search-box" id="search" placeholder="Search by case ID or title..." oninput="filterCases()">
  <select class="sort-select" id="sortBy" onchange="filterCases()">
    <option value="id">Sort: Case ID</option>
    <option value="r1">Sort: ROUGE-1 ↓</option>
    <option value="ot">Sort: OT Score ↓</option>
  </select>
  <button class="btn" id="prevBtn" onclick="changePage(-1)">← Prev</button>
  <span class="page-info" id="pageInfo"></span>
  <button class="btn" id="nextBtn" onclick="changePage(1)">Next →</button>
</div>

<div class="grid" id="grid"></div>
<div class="footer">MHMS Framework &mdash; Optimal Transport Alignment &mdash; CNN/DailyMail Multimodal Dataset</div>

<script>
const ALL_CASES = {cases_json};
const PER_PAGE = 20;
let filtered = [...ALL_CASES];
let page = 0;

function filterCases() {{
  const q = document.getElementById('search').value.toLowerCase();
  const sort = document.getElementById('sortBy').value;
  filtered = ALL_CASES.filter(c => c.id.includes(q) || (c.title||'').toLowerCase().includes(q));
  if (sort === 'r1') filtered.sort((a,b) => (b.r1||0) - (a.r1||0));
  else if (sort === 'ot') filtered.sort((a,b) => b.ot - a.ot);
  else filtered.sort((a,b) => parseInt(a.id) - parseInt(b.id));
  page = 0;
  render();
}}

function changePage(d) {{
  page = Math.max(0, Math.min(page + d, Math.ceil(filtered.length / PER_PAGE) - 1));
  render();
  window.scrollTo({{top: document.querySelector('.controls').offsetTop - 10, behavior: 'smooth'}});
}}

function render() {{
  const start = page * PER_PAGE;
  const slice = filtered.slice(start, start + PER_PAGE);
  const totalPages = Math.ceil(filtered.length / PER_PAGE);
  document.getElementById('pageInfo').textContent = `Page ${{page+1}} of ${{totalPages}} (${{filtered.length}} cases)`;
  document.getElementById('prevBtn').disabled = page === 0;
  document.getElementById('nextBtn').disabled = page >= totalPages - 1;

  const grid = document.getElementById('grid');
  grid.innerHTML = slice.map(c => `
    <div class="case-card">
      <div class="case-header">
        <span class="case-id">#${{c.id}}</span>
        <span class="case-title">${{esc(c.title || 'Untitled')}}</span>
        <span class="ot-badge">OT ${{c.ot.toFixed(6)}}</span>
      </div>
      <div class="case-body">
        <div class="img-col">
          ${{c.img ? `<img src="${{c.img}}" alt="Keyframe">` : '<div class="no-img">No keyframe</div>'}}
        </div>
        <div class="text-col">
          <div><span class="section-label gen">Generated Summary</span></div>
          <div class="section-text">${{esc(c.pred)}}</div>
          <div><span class="section-label gt">Ground Truth</span></div>
          <div class="section-text">${{esc(c.gt)}}</div>
        </div>
      </div>
      <div class="case-metrics">
        ${{c.r1 !== null ? `<span class="metric">R1: <b>${{c.r1}}</b></span>` : ''}}
        ${{c.r2 !== null ? `<span class="metric">R2: <b>${{c.r2}}</b></span>` : ''}}
        ${{c.rL !== null ? `<span class="metric">RL: <b>${{c.rL}}</b></span>` : ''}}
        <span class="metric">Text P: <b>${{c.tp.toFixed(4)}}</b></span>
        <span class="metric">Vid P: <b>${{c.vp.toFixed(4)}}</b></span>
        <span class="metric">Keyframes: <b>${{c.ni}}</b></span>
      </div>
    </div>
  `).join('');
}}

function esc(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}

filterCases();
</script>
</body>
</html>"""
    return html


def main():
    print("Loading cases...")
    cases = load_cases()
    print(f"  Found {len(cases)} complete cases")
    print("Building HTML...")
    html = build_html(cases)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Wrote {OUTPUT_HTML} ({len(html):,} bytes)")

    # Also update the PNG with more cases (8 instead of 4)
    update_png_report(cases[:8])
    print("Done!")


def update_png_report(cases_subset):
    """Generate an updated multimodal_showcase.png with more cases."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import textwrap
        from PIL import Image as PILImage
    except ImportError:
        print("  Skipping PNG (matplotlib/PIL not available)")
        return

    all_figs = []
    for i, c in enumerate(cases_subset):
        wrap_w = 75
        wrapped_pred = textwrap.fill(c["pred"], width=wrap_w)
        wrapped_gt = textwrap.fill(c["gt"], width=wrap_w)
        rouge_line = ""
        if c["rouge"]:
            rouge_line = f"ROUGE-1: {c['rouge']['rouge1']:.2f}  |  ROUGE-2: {c['rouge']['rouge2']:.2f}  |  ROUGE-L: {c['rouge']['rougeL']:.2f}\n"

        text_block = (
            f"CASE #{c['id']}  —  {c['title'][:70]}\n"
            f"{'═' * 50}\n"
            f"{rouge_line}"
            f"OT Match: {c['top_ot']:.6f}  |  Text P: {c['avg_text_prob']:.4f}  |  Vid P: {c['avg_vid_prob']:.4f}\n\n"
            f"GENERATED SUMMARY:\n{'─' * 40}\n{wrapped_pred}\n\n"
            f"GROUND TRUTH:\n{'─' * 40}\n{wrapped_gt}"
        )

        num_lines = text_block.count("\n") + 1
        fig_height = max(num_lines * 0.22 + 1.5, 5.0)
        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(18, fig_height))

        ax_img.axis("off")
        # Find actual image file
        import glob as gl
        imgs = gl.glob(os.path.join(DATA_DIR, c["id"], "*_summary.jpg"))
        if imgs:
            import matplotlib.image as mpimg
            ax_img.imshow(mpimg.imread(imgs[0]))
        else:
            ax_img.text(0.5, 0.5, "No Image", fontsize=14, ha="center", va="center", color="gray")
            ax_img.set_xlim(0, 1); ax_img.set_ylim(0, 1)
        ax_img.set_title(f"Document {c['id']}  (Keyframe)", fontsize=14, fontweight="bold", pad=12)

        ax_text.axis("off"); ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
        ax_text.text(0.02, 0.98, text_block, transform=ax_text.transAxes,
                     fontsize=9, family="monospace", linespacing=1.4,
                     verticalalignment="top", horizontalalignment="left")
        ax_text.set_title("Summary + Metrics", fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout(pad=2.0)
        tmp = f"_tmp_case_{i}.png"
        fig.savefig(tmp, dpi=150, bbox_inches="tight", facecolor="white")
        all_figs.append(tmp)
        plt.close(fig)

    # Stitch
    pil_imgs = [PILImage.open(f) for f in all_figs]
    total_w = max(im.width for im in pil_imgs)
    gap = 20
    total_h = sum(im.height for im in pil_imgs) + gap * (len(pil_imgs) - 1)
    combined = PILImage.new("RGB", (total_w, total_h), (255, 255, 255))
    y = 0
    for im in pil_imgs:
        combined.paste(im, ((total_w - im.width) // 2, y))
        y += im.height + gap
    combined.save("multimodal_showcase.png")
    print(f"  Updated multimodal_showcase.png with {len(cases_subset)} cases")
    for f in all_figs:
        os.remove(f)


if __name__ == "__main__":
    main()
