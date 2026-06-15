#!/usr/bin/env python3
"""
query_shelf.py — CLI entrypoint for Retail Shelf Query
======================================================
Usage:
    python query_shelf.py --image shelf.jpg --query "Coca-Cola"
    python query_shelf.py --image shelf.jpg --query "Pepsi" --output result.jpg
    python query_shelf.py --image shelf.jpg --query "Lay's chips" --det-threshold 0.15
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-vocabulary item verification on retail shelf images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python query_shelf.py --image shelf.jpg --query 'Coca-Cola'\n"
            "  python query_shelf.py --image img.png --query 'Doritos' "
            "--output out.png --det-threshold 0.15\n"
        ),
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input shelf image.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Item name to search for on the shelf.",
    )
    parser.add_argument(
        "--output",
        default="output.jpg",
        help="Path for the annotated output image (default: output.jpg).",
    )
    parser.add_argument(
        "--det-threshold",
        type=float,
        default=0.35,
        help="Grounding DINO detection confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO text matching threshold (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Compute device (default: cuda).",
    )
    parser.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace Grounding DINO model identifier.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose / debug logging.",
    )
    parser.add_argument(
        "--sliced",
        action="store_true",
        help="Use SAHI sliced inference for better small object detection.",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=512,
        help="Slice height/width in pixels for SAHI (default: 512).",
    )
    parser.add_argument(
        "--slice-overlap",
        type=float,
        default=0.2,
        help="Overlap ratio between slices for SAHI (default: 0.2 = 20%%).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- logging setup ----
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- device validation ----
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning(
            "CUDA requested but not available. Falling back to CPU."
        )
        args.device = "cpu"

    if args.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  🖥  GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("  ⚠  Running on CPU — inference will be slower.")

    # ---- run pipeline ----
    from inference import ShelfDetector  # deferred import for faster --help

    t0 = time.perf_counter()
    detector = ShelfDetector(
        model_name=args.model,
        device=args.device,
        det_threshold=args.det_threshold,
        text_threshold=args.text_threshold,
        use_sliced_inference=args.sliced,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_ratio=args.slice_overlap,
    )
    results = detector.run(
        image_path=args.image,
        query=args.query,
        output_path=args.output,
    )
    elapsed = time.perf_counter() - t0

    print(f"  ⏱  Total elapsed: {elapsed:.1f}s")

    if not results:
        sys.exit(1)  # non-zero exit when item not found


if __name__ == "__main__":
    main()
