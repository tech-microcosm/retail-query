"""
inference.py — Open-Vocabulary Shelf Detection Pipeline
========================================================
Combines Grounding DINO zero-shot object detection with EasyOCR text extraction
and difflib/SequenceMatcher string similarity to locate and verify items
on a retail shelf image.

Pipeline stages:
    1. Grounding DINO detects candidate regions for the text query.
    2. EasyOCR reads visible text inside each candidate bounding box.
    3. A combined score (detection confidence + text similarity) is computed.
    4. Bounding boxes are drawn with colour coding (green >= 0.70, orange 0.35-0.70).

Fallback (OCR-first) path:
    When Grounding DINO finds no candidates (common for brand name text queries),
    EasyOCR scans the full image for matching text and builds detections directly
    from OCR bounding boxes, bypassing the detection model entirely.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.ops import nms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single detected region with scoring metadata."""
    box: Tuple[int, int, int, int]      # (x_min, y_min, x_max, y_max)
    det_score: float                     # raw Grounding DINO confidence
    ocr_texts: List[str] = field(default_factory=list)
    text_sim: float = 0.0               # best difflib similarity vs query
    combined_score: float = 0.0         # final blended score


# ---------------------------------------------------------------------------
# Helper: text similarity
# ---------------------------------------------------------------------------

def text_similarity(query: str, candidate: str) -> float:
    """Return a 0-1 similarity score between *query* and *candidate*.

    Handles multi-word queries (e.g. "kalamata olives") by also scoring each
    individual query word against the candidate, then taking the best result.
    This ensures a single OCR token like "Klamaa" still scores well against
    the full query "kalamata olives".
    """
    q = query.lower().strip()
    c = candidate.lower().strip()
    if not q or not c:
        return 0.0

    # Full-string match
    direct_score = difflib.SequenceMatcher(None, q, c).ratio()

    # Substring boost
    if q in c or c in q:
        return max(direct_score, 0.85)

    # Per-word match: score each word in the query against the candidate.
    # Only apply for words long enough that a match is meaningful (>=5 chars),
    # and only promote the score if the per-word match is strong (>=0.75).
    # This avoids short coincidental matches like "alamy" ~ "kalamata".
    words = q.split()
    if len(words) > 1:
        long_words = [w for w in words if len(w) >= 5]
        if long_words:
            per_word_best = max(
                difflib.SequenceMatcher(None, w, c).ratio() for w in long_words
            )
            # Substring boost at word level (exact containment only)
            if any(w in c or c in w for w in long_words):
                per_word_best = max(per_word_best, 0.85)
            # Only use per-word score if it's genuinely strong
            if per_word_best >= 0.75:
                return max(direct_score, per_word_best)

    return direct_score


def best_text_match(query: str, ocr_texts: List[str]) -> float:
    """Return the highest similarity score across all *ocr_texts* fragments."""
    if not ocr_texts:
        return 0.0
    return max(text_similarity(query, t) for t in ocr_texts)


# ---------------------------------------------------------------------------
# Core pipeline class
# ---------------------------------------------------------------------------

class ShelfDetector:
    """End-to-end open-vocabulary shelf item detector.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for Grounding DINO.
    device : str
        ``"cuda"`` or ``"cpu"``.
    det_threshold : float
        Minimum raw Grounding DINO score to keep a candidate box (pre-OCR).
    text_threshold : float
        Minimum text similarity threshold for Grounding DINO.
    ocr_languages : list[str]
        Language codes passed to EasyOCR (default ``["en"]``).
    """

    # Weights for blending detection confidence and OCR text similarity.
    # Detection gets much higher weight because:
    #   1. Grounding DINO already understands text via its language encoder
    #   2. Retail label OCR is often noisy (small fonts, curved bottles, glare)
    DET_WEIGHT: float = 0.75
    OCR_WEIGHT: float = 0.25

    # Colour-code thresholds (adjusted for retail shelf reality)
    HIGH_THRESH: float = 0.70  # Strong match
    MOD_THRESH: float = 0.35   # Moderate match - lowered for practical retail use

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        det_threshold: float = 0.35,
        text_threshold: float = 0.25,
        ocr_languages: Optional[List[str]] = None,
        use_sliced_inference: bool = False,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_ratio: float = 0.2,
    ) -> None:
        self.device = torch.device(device)
        self.det_threshold = det_threshold
        self.text_threshold = text_threshold
        self.use_sliced_inference = use_sliced_inference
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio

        logger.info("Loading Grounding DINO model: %s on %s …", model_name, device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        # Move to GPU (keep fp32 - Grounding DINO's deformable attention doesn't support fp16)
        self.model.to(self.device).eval()
        logger.info("Grounding DINO ready.")

        logger.info("Initialising EasyOCR reader …")
        import easyocr  # lazy import — heavy first-time download
        self.ocr_reader = easyocr.Reader(
            ocr_languages or ["en"],
            gpu=device == "cuda",
        )
        logger.info("EasyOCR ready.")

    # ----- detection ----------------------------------------------------------

    def _detect(
        self, image: Image.Image, query: str
    ) -> List[Detection]:
        """Run Grounding DINO and return raw ``Detection`` objects.
        
        Tries multiple query formulations to improve detection:
        - Original query
        - "package" variant for better shelf item detection
        """
        # Grounding DINO prefers simple text queries (no prefix needed)
        base_query = query.lower().strip()
        
        # Try multiple query formulations for better detection
        query_variants = [
            base_query,
            f"{base_query} package",  # Helps detect packaged items
        ]
        
        all_detections = {}  # Use dict to deduplicate by box coordinates
        
        for text_query in query_variants:
            inputs = self.processor(
                images=image, text=text_query, return_tensors="pt"
            )
            # Move inputs to device (keep fp32 for compatibility)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                threshold=self.det_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

            for box, score in zip(results["boxes"], results["scores"]):
                x_min, y_min, x_max, y_max = [int(coord) for coord in box.tolist()]
                box_key = (x_min, y_min, x_max, y_max)
                
                # Keep highest score for duplicate boxes
                if box_key not in all_detections or score.item() > all_detections[box_key].det_score:
                    all_detections[box_key] = Detection(
                        box=box_key,
                        det_score=float(score.item()),
                    )
        
        detections = list(all_detections.values())
        
        # Apply NMS to remove large boxes that overlap with smaller, tighter boxes
        if len(detections) > 1:
            detections = self._apply_nms(detections, iou_threshold=0.5)
        
        return detections
    
    # Minimum long edge in pixels for reliable OCR.
    # Images smaller than this are upscaled before OCR scanning.
    OCR_MIN_LONG_EDGE: int = 1400

    def _upscale_for_ocr(self, image_np: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (upscaled_image, scale_factor).

        If the image is already large enough, returns the original and scale=1.0.
        Upscaling with LANCZOS gives EasyOCR significantly better character
        recognition on low-resolution shelf photos.
        """
        h, w = image_np.shape[:2]
        long_edge = max(h, w)
        if long_edge >= self.OCR_MIN_LONG_EDGE:
            return image_np, 1.0

        scale = self.OCR_MIN_LONG_EDGE / long_edge
        new_w, new_h = int(w * scale), int(h * scale)
        pil = Image.fromarray(image_np)
        upscaled = np.array(pil.resize((new_w, new_h), Image.LANCZOS))
        logger.info(
            "Image upscaled %dx%d → %dx%d (%.1fx) for OCR.",
            w, h, new_w, new_h, scale,
        )
        return upscaled, scale

    def _preprocess_variants(self, image_np: np.ndarray) -> List[np.ndarray]:
        """Return a list of preprocessed variants of *image_np* for multi-pass OCR.

        Running EasyOCR on multiple contrast/sharpness variants and merging
        results is more reliable than a single pass, especially for:
        - Dark labels with light text (jars, bottles)
        - Low-contrast or pixelated text
        - Curved label surfaces
        """
        variants = [image_np]   # always include the original

        # Variant 2: LAB colour space CLAHE — boosts local contrast while
        # preserving colour so EasyOCR colour cues remain intact
        try:
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            l, a, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
            lab_eq = cv2.merge([clahe.apply(l), a, b_ch])
            variants.append(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB))
        except Exception:
            pass

        # Variant 3: Grayscale + CLAHE + unsharp mask — maximises text edge
        # contrast; helps EasyOCR on small, low-resolution characters
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            clahe_g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            eq = clahe_g.apply(gray)
            blurred = cv2.GaussianBlur(eq, (0, 0), 2)
            sharp = cv2.addWeighted(eq, 1.5, blurred, -0.5, 0)
            variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB))
        except Exception:
            pass

        return variants

    def _detect_ocr_first(
        self, image_np: np.ndarray, query: str
    ) -> List[Detection]:
        """OCR-first fallback: scan the full image with EasyOCR and return
        bounding boxes for every text region that fuzzy-matches the query.

        Works well for brand-name / text queries (e.g. "GOLD", "Natural Cane Sugar")
        where the text is literally printed on the package.
        Automatically upscales low-resolution images before scanning.
        """
        q = query.lower().strip()

        # Upscale if image is too small for reliable OCR
        ocr_img, scale = self._upscale_for_ocr(image_np)

        # Run OCR on multiple preprocessed variants and merge all results.
        # Each variant may surface text that others miss.
        all_raw: List[Tuple] = []
        seen_texts: set = set()
        for variant in self._preprocess_variants(ocr_img):
            for bbox_pts, text, conf in self.ocr_reader.readtext(variant):
                key = (text.lower().strip(), tuple(tuple(p) for p in bbox_pts))
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_raw.append((bbox_pts, text, conf))

        raw = all_raw
        logger.debug("Multi-pass OCR produced %d unique text regions.", len(raw))
        detections: dict = {}

        for (bbox_pts, text, conf) in raw:
            sim = text_similarity(q, text)
            if sim < 0.30:          # slightly looser to catch pixelated text
                continue

            # bbox_pts is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] in upscaled coords
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            # Map coordinates back to original image space
            x1 = int(x1 / scale); y1 = int(y1 / scale)
            x2 = int(x2 / scale); y2 = int(y2 / scale)

            # Expand box outward to capture the full package label region
            h, w = image_np.shape[:2]
            pad_x = max(int((x2 - x1) * 0.5), 20)
            pad_y = max(int((y2 - y1) * 1.5), 20)
            x1e = max(0, x1 - pad_x)
            y1e = max(0, y1 - pad_y)
            x2e = min(w, x2 + pad_x)
            y2e = min(h, y2 + pad_y)

            box_key = (x1e, y1e, x2e, y2e)
            det_proxy = conf * 0.6   # scale OCR confidence into detection range
            if box_key not in detections or det_proxy > detections[box_key].det_score:
                # In OCR-first mode, text similarity is the primary signal.
                # Use inverted weights vs the normal pipeline: sim drives the score.
                combined = 0.35 * det_proxy + 0.65 * sim
                detections[box_key] = Detection(
                    box=box_key,
                    det_score=det_proxy,
                    ocr_texts=[text],
                    text_sim=sim,
                    combined_score=combined,
                )

        candidates = list(detections.values())
        if len(candidates) > 1:
            candidates = self._apply_nms(candidates, iou_threshold=0.3)
        return candidates

    def _apply_nms(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Apply Non-Maximum Suppression to prefer tighter bounding boxes.
        
        When multiple boxes overlap significantly, keep the one with higher confidence.
        This helps avoid large boxes that cover multiple items.
        """
        if not detections:
            return []
        
        boxes = torch.tensor([d.box for d in detections], dtype=torch.float32)
        scores = torch.tensor([d.det_score for d in detections], dtype=torch.float32)
        
        # Convert from (x1, y1, x2, y2) format - already in correct format
        keep_indices = nms(boxes, scores, iou_threshold)
        
        return [detections[i] for i in keep_indices.tolist()]
    
    def _detect_sliced(
        self, image: Image.Image, query: str
    ) -> List[Detection]:
        """Run Grounding DINO with manual sliced inference.
        
        Divides the image into overlapping patches, runs detection on each patch,
        and merges results. This significantly improves detection of small objects.
        """
        from sahi.slicing import slice_image
        
        img_width, img_height = image.size
        all_detections = {}  # Use dict to track unique detections
        
        # Calculate slice parameters
        slice_h = self.slice_height
        slice_w = self.slice_width
        overlap_h = int(slice_h * self.overlap_ratio)
        overlap_w = int(slice_w * self.overlap_ratio)
        
        # Generate slices
        slices = slice_image(
            image=np.array(image),
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
        )
        
        logger.info(
            f"Processing {len(slices.images)} image slices "
            f"({slice_h}x{slice_w} with {self.overlap_ratio*100:.0f}% overlap)"
        )
        
        # Process each slice
        for slice_img, shift in zip(slices.images, slices.starting_pixels):
            # Convert slice to PIL Image
            pil_slice = Image.fromarray(slice_img)
            
            # Run detection on this slice
            text_query = query.lower().strip()
            inputs = self.processor(
                images=pil_slice, text=text_query, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                threshold=self.det_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil_slice.size[::-1]],
            )[0]
            
            # Transform coordinates back to original image space
            shift_x, shift_y = shift
            for box, score in zip(results["boxes"], results["scores"]):
                x1, y1, x2, y2 = box.tolist()
                
                # Apply shift to get coordinates in original image
                global_x1 = int(x1 + shift_x)
                global_y1 = int(y1 + shift_y)
                global_x2 = int(x2 + shift_x)
                global_y2 = int(y2 + shift_y)
                
                # Clamp to image boundaries
                global_x1 = max(0, min(global_x1, img_width))
                global_y1 = max(0, min(global_y1, img_height))
                global_x2 = max(0, min(global_x2, img_width))
                global_y2 = max(0, min(global_y2, img_height))
                
                box_key = (global_x1, global_y1, global_x2, global_y2)
                score_val = float(score.item())
                
                # Keep highest score for duplicate/overlapping boxes
                if box_key not in all_detections or score_val > all_detections[box_key].det_score:
                    all_detections[box_key] = Detection(
                        box=box_key,
                        det_score=score_val,
                    )
        
        detections = list(all_detections.values())
        
        # Apply NMS to merge overlapping detections from different slices
        if len(detections) > 1:
            detections = self._apply_nms(detections, iou_threshold=0.3)
        
        return detections

    # ----- OCR refinement -----------------------------------------------------

    def _ocr_refine(
        self, image_np: np.ndarray, detections: List[Detection], query: str
    ) -> List[Detection]:
        """Run EasyOCR inside each bounding box and compute combined scores.

        Detections that already have a combined_score set (from _detect_ocr_first)
        are skipped — their scores are already text-similarity driven and accurate.
        """
        h, w = image_np.shape[:2]
        for det in detections:
            # Skip OCR-first detections — already scored with correct weights
            if det.combined_score > 0.0 and det.ocr_texts:
                continue

            x1, y1, x2, y2 = det.box
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Run multi-pass OCR on the crop for better accuracy
            all_texts: List[str] = []
            seen: set = set()
            for variant in self._preprocess_variants(crop):
                for t in self.ocr_reader.readtext(variant, detail=0):
                    t = t.strip()
                    if t and t.lower() not in seen:
                        seen.add(t.lower())
                        all_texts.append(t)

            det.ocr_texts = all_texts
            det.text_sim = best_text_match(query, det.ocr_texts)
            det.combined_score = (
                self.DET_WEIGHT * det.det_score
                + self.OCR_WEIGHT * det.text_sim
            )
        return detections

    # ----- visualisation ------------------------------------------------------

    @staticmethod
    def _draw(
        image_np: np.ndarray,
        detections: List[Detection],
        query: str,
    ) -> np.ndarray:
        """Draw bounding boxes on the image (BGR for OpenCV).

        Priority rules:
        - If any HIGH-confidence (green) detections exist, draw ONLY those.
        - If no green detections exist, fall back to drawing MODERATE (orange) ones.
        - Green boxes use a thick outline (4 px) for clear identification.
        - Orange boxes use a thinner outline (2 px) as a fallback indicator.
        """
        annotated = image_np.copy()
        GREEN  = (0, 210, 0)
        ORANGE = (0, 140, 255)
        FONT   = cv2.FONT_HERSHEY_SIMPLEX

        # Scale thickness and font to image size so boxes are visible on any
        # resolution — roughly 0.4% of the shorter dimension, minimum 2 px.
        img_h, img_w = annotated.shape[:2]
        scale_ref   = min(img_h, img_w)
        thickness_green  = max(2, int(scale_ref * 0.006))
        thickness_orange = max(2, int(scale_ref * 0.003))
        font_scale  = 0.45
        font_thick  = 1

        qualified = [
            d for d in detections if d.combined_score >= ShelfDetector.MOD_THRESH
        ]
        high = [d for d in qualified if d.combined_score >= ShelfDetector.HIGH_THRESH]

        # Show only green boxes when any exist; otherwise fall back to orange
        to_draw   = high if high else qualified
        colour    = GREEN if high else ORANGE
        thickness = thickness_green if high else thickness_orange

        for det in to_draw:
            x1, y1, x2, y2 = det.box
            score = det.combined_score
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

            # Label
            ocr_tag = " | ".join(det.ocr_texts[:3]) if det.ocr_texts else "\u2014"
            label = f"{query} [{score:.2f}] OCR: {ocr_tag}"
            (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, font_thick)
            cv2.rectangle(
                annotated,
                (x1, max(y1 - th - 8, 0)),
                (x1 + tw + 4, y1),
                colour,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(y1 - 4, th + 2)),
                FONT,
                font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA,
            )
        return annotated

    # ----- public API ---------------------------------------------------------

    def run(
        self,
        image_path: str,
        query: str,
        output_path: str = "output.jpg",
    ) -> List[Detection]:
        """Full pipeline: detect → OCR refine → score → annotate → save.

        Parameters
        ----------
        image_path : str
            Path to the input shelf image.
        query : str
            Item name to search for (e.g. ``"Coca-Cola"``).
        output_path : str
            Where to write the annotated image.

        Returns
        -------
        list[Detection]
            All detections that cleared the moderate threshold (≥ 0.35).
        """
        if self.use_sliced_inference:
            logger.info(
                "Processing with SAHI sliced inference: image=%s  query='%s'  "
                "slice_size=%dx%d overlap=%.1f%%",
                image_path, query, self.slice_height, self.slice_width, 
                self.overlap_ratio * 100
            )
        else:
            logger.info("Processing: image=%s  query='%s'", image_path, query)

        # Load image
        pil_image = Image.open(image_path).convert("RGB")
        image_np = np.array(pil_image)              # RGB
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Stage 1: Grounding DINO detection
        img_h, img_w = image_np.shape[:2]
        use_sliced = self.use_sliced_inference and max(img_h, img_w) >= self.slice_height * 2
        if use_sliced:
            detections = self._detect_sliced(pil_image, query)
            logger.info(
                "Grounding DINO (sliced) returned %d candidate(s).", 
                len(detections)
            )
        else:
            if self.use_sliced_inference and not use_sliced:
                logger.info(
                    "Image too small for meaningful slicing (%dx%d) — "
                    "running standard detection instead.",
                    img_w, img_h,
                )
            detections = self._detect(pil_image, query)
            logger.info("Grounding DINO returned %d candidate(s).", len(detections))

        # Always run OCR-first scan as a complement to Grounding DINO.
        # DINO finds visual object regions; OCR finds exact text matches.
        # Merging both gives the best recall on brand-name queries.
        logger.info("Running OCR-first scan in parallel with DINO results.")
        ocr_detections = self._detect_ocr_first(image_np, query)
        if ocr_detections:
            logger.info("OCR-first scan found %d candidate(s).", len(ocr_detections))
            detections = detections + ocr_detections

        if not detections:
            print(
                f"\n  ✗  Item '{query}' was NOT found on the shelf.\n"
                f"      Grounding DINO and OCR scan both found no candidates.\n"
            )
            cv2.imwrite(output_path, image_bgr)
            return []

        # Stage 2: OCR refinement + combined scoring
        # OCR-first detections already have combined_score set; _ocr_refine will
        # re-run OCR inside the bounding boxes for DINO detections that don't yet.
        detections = self._ocr_refine(image_np, detections, query)

        # Sort by combined score descending
        detections.sort(key=lambda d: d.combined_score, reverse=True)

        # Filter to those clearing the moderate threshold
        qualified = [d for d in detections if d.combined_score >= self.MOD_THRESH]

        if not qualified:
            print(
                f"\n  ✗  Item '{query}' was NOT found on the shelf.\n"
                f"      {len(detections)} candidate(s) were evaluated, but "
                f"none reached the minimum combined score of "
                f"{self.MOD_THRESH:.2f}.\n"
            )
            # Debug: show top detection scores
            if detections:
                print("  📊  Top detection scores (for debugging):")
                for i, det in enumerate(detections[:3], 1):
                    ocr_preview = ", ".join(det.ocr_texts[:3]) if det.ocr_texts else "—"
                    print(
                        f"      #{i}  combined={det.combined_score:.3f} "
                        f"(det={det.det_score:.3f}, txt_sim={det.text_sim:.3f})  "
                        f"OCR: '{ocr_preview}'"
                    )
                print()
            # Still save an unannotated copy for reference
            cv2.imwrite(output_path, image_bgr)
            return []

        # Print results
        print(f"\n  ✓  Results for '{query}':")
        for i, det in enumerate(qualified, 1):
            tier = "HIGH" if det.combined_score >= self.HIGH_THRESH else "MODERATE"
            colour_label = "green" if tier == "HIGH" else "orange"
            ocr_tag = ", ".join(det.ocr_texts[:3]) if det.ocr_texts else "—"
            print(
                f"      #{i}  score={det.combined_score:.2f} "
                f"(det={det.det_score:.2f}, txt={det.text_sim:.2f})  "
                f"[{tier} → {colour_label} box]  "
                f"OCR: '{ocr_tag}'"
            )
        print()

        # Stage 3: Draw and save
        annotated = self._draw(image_bgr, qualified, query)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, annotated)
        logger.info("Annotated image saved → %s", output_path)
        print(f"  💾  Annotated image saved to: {output_path}")

        # Stage 4: Save zoomed-in crop of the green-box region for blog use.
        # The crop covers the union of all high-confidence boxes with padding.
        high_dets = [d for d in qualified if d.combined_score >= self.HIGH_THRESH]
        if high_dets:
            xs1 = [d.box[0] for d in high_dets]
            ys1 = [d.box[1] for d in high_dets]
            xs2 = [d.box[2] for d in high_dets]
            ys2 = [d.box[3] for d in high_dets]
            img_h, img_w = image_bgr.shape[:2]
            pad = max(40, int(min(img_h, img_w) * 0.04))
            cx1 = max(0, min(xs1) - pad)
            cy1 = max(0, min(ys1) - pad)
            cx2 = min(img_w, max(xs2) + pad)
            cy2 = min(img_h, max(ys2) + pad)
            zoom_crop = annotated[cy1:cy2, cx1:cx2]
            stem, ext = Path(output_path).stem, Path(output_path).suffix
            zoom_path = str(Path(output_path).parent / f"{stem}_zoom{ext}")
            cv2.imwrite(zoom_path, zoom_crop)
            logger.info("Zoomed crop saved → %s", zoom_path)
            print(f"  🔍  Zoomed crop saved to:   {zoom_path}")
        print()

        return qualified
