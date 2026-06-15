# Retail Shelf Query — Open-Vocabulary Item Verification

An open-vocabulary item verification tool for grocery/retail shelves.
Given an image of a shelf and a text query (item name), the tool locates matching
items using **Grounding DINO** zero-shot detection, refines results with **EasyOCR**
text extraction, and draws colour-coded bounding boxes on the output image.

## Colour-coded scoring

| Colour | Meaning | Combined score |
|--------|---------|----------------|
| 🟩 Green | High-confidence match | ≥ 0.70 |
| 🟧 Orange | Moderate / partial match | 0.35 – 0.70 |
| — | Not shown (filtered out) | < 0.35 |

Combined score = `0.75 × detection_conf + 0.25 × ocr_similarity`  
(Detection weighted higher since Grounding DINO already encodes text understanding)

If **no** detection clears the 0.35 threshold, a clear console message is printed.

## SAHI Sliced Inference (for small objects)

For dense shelf scenes with many small packages, use **`--sliced`** mode:

```bash
python query_shelf.py --image shelf.jpg --query "item" --sliced
```

**How it works:**
1. Divides image into overlapping 512×512 patches (20% overlap by default)
2. Runs Grounding DINO on each patch independently
3. Merges results with NMS to remove duplicates

**Benefits:**
- Improves detection of small text/packages
- Better for high-resolution images
- Each patch gets more model attention

**Trade-offs:**
- ~2x slower (processes multiple patches)
- May still struggle with very small or low-contrast text

## Quick start

```bash
# 1. Create & activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (CUDA wheels recommended)
pip install -r requirements.txt

# 3. Run
python query_shelf.py --image shelf.jpg --query "Coca-Cola" --output result.jpg
```

## CLI arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--image` | ✅ | Path to the input shelf image |
| `--query` | ✅ | Item name to search for |
| `--output` | ❌ | Path for the annotated output image (default: `output.jpg`) |
| `--det-threshold` | ❌ | Grounding DINO detection confidence threshold (default: `0.35`) |
| `--text-threshold` | ❌ | Grounding DINO text matching threshold (default: `0.25`) |
| `--device` | ❌ | Force device (`cuda` / `cpu`, default: `cuda`) |
| `--sliced` | ❌ | Enable SAHI sliced inference for better small object detection |
| `--slice-size` | ❌ | Slice height/width in pixels (default: `512`) |
| `--slice-overlap` | ❌ | Overlap ratio between slices (default: `0.2` = 20%) |

## Project structure

```
retail-query/
├── query_shelf.py      # CLI entrypoint
├── inference.py         # Detection + OCR + scoring pipeline
├── requirements.txt
└── README.md
```

## Hardware requirements

Designed for an **NVIDIA RTX 4050** (or any CUDA-capable GPU).
The pipeline runs in **fp16 mixed-precision** to keep VRAM usage low (~2-3 GB).

## Test images

Recommended open datasets for testing:
- **Grocery Dataset** — https://github.com/gulvarol/grocerydataset (~354 shelf images)
- **Roboflow Retail datasets** — https://universe.roboflow.com/browse/retail
- **HuggingFace grocery-shelves** — https://huggingface.co/datasets/UniDataPro/grocery-shelves
