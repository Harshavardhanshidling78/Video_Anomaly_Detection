import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Generator, Tuple, Optional
from src.config import DEVICE, ANOMALY_THRESHOLD, CHECKPOINT_DIR, CLIP_LEN, FRAME_STRIDE, RECURRENT_TYPE

from src.preprocessing.video_processor import FastVideoProcessor
from src.models.anomaly_classifier import VideoAnomalyDetectorModel
from src.utils.database import AnomalyDatabase

# Optional Ultralytics YOLO for Real-Time CCTV Weapon/Object Threat Scanning
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

class VideoAnomalyDetector:
    """
    Enterprise Dual-Stream Video Anomaly Detection System:
    - Stream 1: Vision Transformer (ViT-B/16) + LSTM Sequence Temporal Anomaly Model (fights, falls, chaos)
    - Stream 2: High-Resolution Spatial Threat & Weapon Detector (CCTV handguns, knives, blades)
    - Vectorized C++ OpenCV Ingestion with aspect-ratio preserving letterboxing
    - Normalized ~10 FPS temporal sampling matching training motion cadence
    - Sustained Anomaly Verification (eliminates 1-frame false-positive spikes)
    - EMA (Exponential Moving Average) temporal noise filtering
    - Thread-safe SQLite / PostgreSQL dual-persistence
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        weapon_model_path: Optional[str] = None,
        spatial_mode: str = "vit",
        threshold: float = ANOMALY_THRESHOLD,
        device: str = DEVICE,
        ema_alpha: float = 0.35
    ):
        self.device = device
        self.threshold = threshold
        self.spatial_mode = spatial_mode
        self.ema_alpha = ema_alpha
        self.processor = FastVideoProcessor()
        self.db = AnomalyDatabase()
        self._feature_cache = {}
        
        # 1. Initialize Deep Learning Temporal Model (ViT + LSTM/GRU)
        has_checkpoint = bool(model_path and Path(model_path).exists())
        self.model = VideoAnomalyDetectorModel(
            spatial_mode=spatial_mode,
            pretrained_vit=not has_checkpoint,
            threshold=threshold
        ).to(self.device)
        self.model.eval()

        if has_checkpoint:
            try:
                checkpoint_obj = torch.load(model_path, map_location=self.device)
                sd = checkpoint_obj["state_dict"] if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj else checkpoint_obj
                ckpt_mode = checkpoint_obj.get("spatial_mode", spatial_mode) if isinstance(checkpoint_obj, dict) else spatial_mode
                ckpt_cell = checkpoint_obj.get("cell_type", RECURRENT_TYPE) if isinstance(checkpoint_obj, dict) else RECURRENT_TYPE
                # Auto-detect bidirectional from weight dimensions or dict flag
                ckpt_bidi = checkpoint_obj.get("bidirectional", None) if isinstance(checkpoint_obj, dict) else None
                if ckpt_bidi is None:
                    head_w = sd.get("classifier_head.0.weight", None)
                    ckpt_bidi = (head_w is not None and head_w.shape[1] > TEMPORAL_HIDDEN_DIM)
                
                ckpt_thresh = checkpoint_obj.get("optimal_threshold", threshold) if isinstance(checkpoint_obj, dict) else threshold
                self.threshold = ckpt_thresh
                self.spatial_mode = ckpt_mode
                del self.model
                import gc
                gc.collect()
                self.model = VideoAnomalyDetectorModel(
                    spatial_mode=ckpt_mode,
                    pretrained_vit=False,
                    cell_type=ckpt_cell,
                    threshold=ckpt_thresh,
                    bidirectional=ckpt_bidi
                ).to(self.device)
                self.model.load_state_dict(sd)
                self.model.eval()
                print(f"[Info] Loaded trained model checkpoint from {model_path} (Bi-LSTM={ckpt_bidi}, Threshold={ckpt_thresh:.2f})")
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint ({e}). Using baseline initialized weights.")

        # 2. Initialize Spatial Threat & Weapon Detector (YOLOv8 CCTV Weapon Model)
        self.weapon_model = None
        if ULTRALYTICS_AVAILABLE:
            if not weapon_model_path:
                w_candidate = CHECKPOINT_DIR / "weapon_yolo" / "best.pt"
                if not w_candidate.exists():
                    w_candidate = CHECKPOINT_DIR / "yolov8n.pt"
                weapon_model_path = str(w_candidate) if w_candidate.exists() else None

            if weapon_model_path and Path(weapon_model_path).exists():
                try:
                    self.weapon_model = YOLO(weapon_model_path)
                    print(f"[Info] Dual-Stream Weapon Detector active: loaded {Path(weapon_model_path).name}")
                except Exception as e:
                    print(f"[Warning] Could not initialize weapon detector: {e}")

            # 3. Initialize General Multi-Class Object Detector (YOLOv8 COCO for Negative Vehicle & Context Veto)
            coco_candidate = CHECKPOINT_DIR / "yolov8n.pt"
            if coco_candidate.exists():
                try:
                    self.coco_model = YOLO(str(coco_candidate))
                    print(f"[Info] General Context & Vehicle Detector active: loaded {coco_candidate.name}")
                except Exception as e:
                    self.coco_model = None
                    print(f"[Warning] Could not initialize COCO detector: {e}")
            else:
                self.coco_model = None


    def set_threshold(self, threshold: float):
        self.threshold = threshold
        self.model.threshold = threshold

    def set_spatial_mode(self, spatial_mode: str):
        if self.spatial_mode != spatial_mode:
            self.spatial_mode = spatial_mode
            self.model = VideoAnomalyDetectorModel(spatial_mode=spatial_mode, threshold=self.threshold).to(self.device)
            self.model.eval()
            self._feature_cache.clear()

    def _apply_ema_smoothing(self, raw_scores: List[float]) -> List[float]:
        """
        Applies Exponential Moving Average (EMA) filtering to remove single-frame noise flicker.
        """
        if not raw_scores:
            return []
        smoothed = []
        curr = raw_scores[0]
        for score in raw_scores:
            curr = self.ema_alpha * score + (1.0 - self.ema_alpha) * curr
            smoothed.append(float(curr))
        return smoothed

    @staticmethod
    def _box_distance(b1: List[float], b2: List[float]) -> float:
        """Compute Euclidean distance between centers of two boxes."""
        cx1 = (b1[0] + b1[2]) / 2.0
        cy1 = (b1[1] + b1[3]) / 2.0
        cx2 = (b2[0] + b2[2]) / 2.0
        cy2 = (b2[1] + b2[3]) / 2.0
        return float(((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5)

    @staticmethod
    def _box_iou(b1: List[float], b2: List[float]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        return float(inter / (area1 + area2 - inter))

    @staticmethod
    def _is_vehicle_overlap(
        w_box: List[float],
        vehicle_boxes: List[List[float]],
        iou_thresh: float = 0.15,
        area_overlap_thresh: float = 0.25
    ) -> bool:
        """
        Determines if a candidate weapon bounding box overlaps with or is part of a vehicle.
        Rejects vehicles (cars, motorcycles, bicycles, buses, trucks) being falsely flagged as weapons.
        """
        if not vehicle_boxes:
            return False
        wx1, wy1, wx2, wy2 = w_box
        w_w = max(1e-3, wx2 - wx1)
        w_h = max(1e-3, wy2 - wy1)
        w_area = w_w * w_h
        w_cx = (wx1 + wx2) / 2.0
        w_cy = (wy1 + wy2) / 2.0

        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            # Check 1: Weapon center inside vehicle box
            if vx1 <= w_cx <= vx2 and vy1 <= w_cy <= vy2:
                return True

            # Check 2: Intersection Area / IoU
            ix1 = max(wx1, vx1)
            iy1 = max(wy1, vy1)
            ix2 = min(wx2, vx2)
            iy2 = min(wy2, vy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                v_area = max(1.0, (vx2 - vx1) * (vy2 - vy1))
                iou = inter / (w_area + v_area - inter)
                if (inter / w_area) >= area_overlap_thresh or iou >= iou_thresh:
                    return True
        return False

    @staticmethod
    def _is_human_hand_or_limb(
        crop: np.ndarray,
        skin_thresh: float = 0.35,
        edge_thresh: float = 0.14
    ) -> bool:
        """
        Determines if a candidate bounding box is an anatomical human hand, fist, or arm
        rather than a fabricated weapon (knife, blade, firearm, bat).
        
        Weapons have high metallic/matte edge density, dark barrels, or reflective steel.
        Human hands/fists have high skin-tone chromatic saturation and smooth tissue transitions.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return False

        try:
            # 1. YCrCb Skin Color Detection (illumination-invariant skin chroma)
            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(
                ycrcb,
                np.array([0, 133, 77], dtype=np.uint8),
                np.array([255, 173, 127], dtype=np.uint8)
            )
            skin_ratio_ycrcb = np.count_nonzero(skin_mask) / max(1, skin_mask.size)

            # 2. HSV Skin Color Detection (warm peach to dark skin hues)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(
                hsv,
                np.array([0, 25, 40], dtype=np.uint8),
                np.array([25, 255, 255], dtype=np.uint8)
            )
            skin_ratio_hsv = np.count_nonzero(hsv_mask) / max(1, hsv_mask.size)

            skin_ratio = max(skin_ratio_ycrcb, skin_ratio_hsv)

            # 3. High-Frequency Edge Contrast Density
            # Real weapons (guns, blades) have rigid metallic contours and sharp edges.
            # Hands and arms have soft texture and low edge density.
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            edge_density = np.count_nonzero(edges) / max(1, edges.size)

            # If predominantly skin tissue without weapon edge profile -> Reject as hand
            if skin_ratio >= skin_thresh and edge_density < edge_thresh:
                return True

            # If overwhelmingly skin tissue (> 55%) -> Definitely a hand/limb
            if skin_ratio > 0.55:
                return True

        except Exception:
            return False

        return False

    @staticmethod
    def _is_person_body_overlap(
        w_box: List[float],
        person_boxes: List[List[float]],
        iou_thresh: float = 0.50,
        area_overlap_thresh: float = 0.65
    ) -> bool:
        """
        Determines if a candidate weapon box is actually a person's entire body.
        Rejects human actors being misclassified as weapons.
        """
        if not person_boxes:
            return False
        wx1, wy1, wx2, wy2 = w_box
        w_area = max(1.0, (wx2 - wx1) * (wy2 - wy1))

        for px1, py1, px2, py2 in person_boxes:
            p_area = max(1.0, (px2 - px1) * (py2 - py1))
            ix1 = max(wx1, px1)
            iy1 = max(wy1, py1)
            ix2 = min(wx2, px2)
            iy2 = min(wy2, py2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                iou = inter / (w_area + p_area - inter)
                if iou >= iou_thresh or (inter / w_area) >= area_overlap_thresh:
                    return True
        return False

    def localize_anomaly_region(
        self,
        curr_frame: np.ndarray,
        ref_frame: Optional[np.ndarray] = None,
        person_boxes: Optional[List[List[float]]] = None,
        weapon_boxes: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Pinpoints the spatial Region of Interest (ROI) of anomalous action inside the frame:
        Surveillance Principle: Anomalies are caused by human actors and handheld threats,
        NEVER by metallic fixtures, overhead bus handrails, ceiling poles, or window glare.
        
        Priority Cascade:
        1. Confirmed Weapon Threat (linked to armed actor or handheld threat location)
        2. Human Actor-Centric Localization:
           - Multiple active people grappling / fighting -> Merged Altercation Envelope
           - Most active person -> Primary Anomalous Actor
        3. Strict Metallic / Ceiling Rejection:
           - Filter out overhead rails (top 35% without vertical extension)
           - Filter out extreme aspect ratios (horizontal pipes / vertical poles)
           - Return middle/lower interaction zone
        4. Fallback: Safe central human-height interaction zone
        """
        if curr_frame is None or curr_frame.size == 0:
            return None

        h, w = curr_frame.shape[:2]

        # Compute dense spatial difference between frames if reference frame available
        diff = None
        if ref_frame is not None and ref_frame.shape == curr_frame.shape:
            try:
                g_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                g_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(g_curr, g_ref)
            except Exception:
                diff = None

        # ── PRIORITY 1: Confirmed Weapon Threat Association
        if weapon_boxes and len(weapon_boxes) > 0:
            top_w = weapon_boxes[0]
            w_box = [int(v) for v in top_w["box"]]
            w_label = top_w.get("label", "WEAPON").upper()

            # If an actor is near the weapon, anchor to the armed actor
            if person_boxes:
                for pbox in person_boxes:
                    px1, py1, px2, py2 = [int(v) for v in pbox]
                    if self._box_distance(w_box, [px1, py1, px2, py2]) <= 0.28 * max(w, h):
                        return {
                            "box": [px1, py1, px2, py2],
                            "label": f"ARMED THREAT ACTOR ({w_label})",
                            "type": "armed_actor"
                        }
            # Otherwise return confirmed weapon bounding box
            return {
                "box": w_box,
                "label": f"CRITICAL THREAT: {w_label}",
                "type": "weapon"
            }

        # ── PRIORITY 2: Human Actor-Centric Localization
        # Pinpoint the actual human actor who is causing the anomaly, not metallic background
        if person_boxes and len(person_boxes) > 0:
            scored_persons = []
            for pb in person_boxes:
                px1, py1, px2, py2 = [int(v) for v in pb]
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(w, px2), min(h, py2)
                p_area = max(1, (px2 - px1) * (py2 - py1))

                # Skip tiny noise boxes (< 1% frame area)
                if p_area < max(600, int(0.012 * w * h)):
                    continue

                motion_energy = 0.0
                if diff is not None and (px2 > px1) and (py2 > py1):
                    crop_diff = diff[py1:py2, px1:px2]
                    if crop_diff.size > 0:
                        motion_energy = float(np.mean(crop_diff))

                scored_persons.append({
                    "box": [px1, py1, px2, py2],
                    "motion": motion_energy,
                    "area": p_area
                })

            if scored_persons:
                # Rank actors by motion activity
                scored_persons.sort(key=lambda x: x["motion"], reverse=True)

                # Check for Altercation / Fight / Grappling (Multiple actors interacting)
                if len(scored_persons) >= 2:
                    p1 = scored_persons[0]["box"]
                    p2 = scored_persons[1]["box"]
                    iou = self._box_iou(p1, p2)
                    dist = self._box_distance(p1, p2)
                    # If two top active actors are close or overlapping, merge their altercation
                    if iou > 0.05 or dist <= 0.30 * max(w, h):
                        merged_box = [
                            max(0, min(p1[0], p2[0])),
                            max(0, min(p1[1], p2[1])),
                            min(w, max(p1[2], p2[2])),
                            min(h, max(p1[3], p2[3]))
                        ]
                        return {
                            "box": merged_box,
                            "label": "ANOMALOUS ACTORS (ALTERCATION)",
                            "type": "altercation"
                        }

                # Single primary anomalous actor
                best_actor = scored_persons[0]
                return {
                    "box": best_actor["box"],
                    "label": "ANOMALOUS ACTOR HERE",
                    "type": "person"
                }

        # ── PRIORITY 3: Motion Saliency with STRICT Metal & Ceiling Rejection
        # Overhead bus handrails, ceiling poles, and window glare must be rejected!
        if diff is not None:
            try:
                blur = cv2.GaussianBlur(diff, (7, 7), 0)
                _, thresh = cv2.threshold(blur, 22, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                dilated = cv2.dilate(thresh, kernel, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                valid_contours = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 400:
                        continue
                    bx, by, bw, bh = cv2.boundingRect(c)

                    # 1. Reject full-screen global jitter (> 75% frame)
                    if bw * bh > 0.75 * (w * h):
                        continue

                    # 2. STRICT CEILING & OVERHEAD METAL RAIL REJECTION:
                    # Contours in the top 35% of the frame without lower body extension
                    # are overhead luggage racks, ceiling handrails, or window glare.
                    if (by <= 0.06 * h) and (by + bh <= 0.40 * h):
                        continue  # Overhead rail / ceiling glare
                    if (by + bh / 2.0) < 0.25 * h:
                        continue  # Top quarter ceiling fixture

                    # 3. METALLIC PIPE & HANDRAIL ASPECT RATIO REJECTION:
                    # Long horizontal tubing (w/h > 3.2) or long vertical pole (h/w > 4.5)
                    aspect_h = bw / max(1e-3, bh)
                    aspect_v = bh / max(1e-3, bw)
                    if aspect_h > 3.2:
                        continue  # Horizontal handrail / pipe
                    if aspect_v > 4.5:
                        continue  # Vertical support pole

                    # Saliency score weighted by lower/middle interaction zone
                    interaction_weight = 1.0 + 0.6 * ((by + bh / 2.0) / h)
                    score_metric = area * interaction_weight
                    valid_contours.append((bx, by, bx + bw, by + bh, score_metric))

                if valid_contours:
                    valid_contours.sort(key=lambda x: x[4], reverse=True)
                    top = valid_contours[0]
                    pad_x = int((top[2] - top[0]) * 0.10)
                    pad_y = int((top[3] - top[1]) * 0.10)
                    return {
                        "box": [
                            max(0, top[0] - pad_x),
                            max(0, top[1] - pad_y),
                            min(w, top[2] + pad_x),
                            min(h, top[3] + pad_y)
                        ],
                        "label": "ANOMALY ACTIVITY ZONE",
                        "type": "motion"
                    }
            except Exception:
                pass

        # ── PRIORITY 4: Safe Body-Height Interaction Zone (Never ceiling handrails)
        return {
            "box": [int(0.15 * w), int(0.25 * h), int(0.85 * w), int(0.85 * h)],
            "label": "ANOMALY ACTIVITY ZONE",
            "type": "zone"
        }

    def draw_annotation(
        self,
        frame: np.ndarray,
        score: float,
        is_anomalous: bool,
        frame_idx: int,
        weapon_boxes: Optional[List[Dict[str, Any]]] = None,
        anomaly_roi: Optional[Dict[str, Any]] = None,
        max_display_width: int = 640
    ) -> np.ndarray:
        """
        Fast visual banner & bounding box telemetry overlay onto video frame.
        Highlights:
        - Critical threat red boxes for detected weapons
        - Spatial localization amber target reticles marking where the anomaly is located
        - Top telemetry verdict banner and progress bar
        """
        if frame is None or frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        if w > max_display_width:
            new_h = int(h * (max_display_width / w))
            scale_x = max_display_width / w
            scale_y = new_h / h
            annotated = cv2.resize(frame, (max_display_width, new_h), interpolation=cv2.INTER_AREA)
            h, w = annotated.shape[:2]
        else:
            scale_x, scale_y = 1.0, 1.0
            annotated = frame.copy()

        has_weapon = bool(weapon_boxes and len(weapon_boxes) > 0)

        # 1. Draw Threat Bounding Boxes for Detected Weapons
        if has_weapon:
            for wbox in weapon_boxes:
                x1 = int(wbox["box"][0] * scale_x)
                y1 = int(wbox["box"][1] * scale_y)
                x2 = int(wbox["box"][2] * scale_x)
                y2 = int(wbox["box"][3] * scale_y)
                label = wbox.get("label", "WEAPON").upper()
                conf = wbox.get("confidence", 0.0)

                # High-visibility pulsing red threat box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                tag = f"CRITICAL THREAT: {label} ({conf*100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1), (0, 0, 220), -1)
                cv2.putText(annotated, tag, (x1 + 4, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw Spatial Anomaly Bounding Box (Pinpointing WHERE the Anomaly Occurred)
        if is_anomalous and anomaly_roi and "box" in anomaly_roi:
            abox = anomaly_roi["box"]
            ax1 = int(abox[0] * scale_x)
            ay1 = int(abox[1] * scale_y)
            ax2 = int(abox[2] * scale_x)
            ay2 = int(abox[3] * scale_y)
            a_label = anomaly_roi.get("label", "ANOMALY LOCATED HERE").upper()

            # High-visibility amber/orange target bounding box
            box_color = (0, 140, 255) # Bright Amber in BGR
            cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), box_color, 2)
            
            # Draw targeting corner brackets for high-tech CCTV surveillance visual
            corner_len = max(8, min(24, (ax2 - ax1) // 4, (ay2 - ay1) // 4))
            # Top-left
            cv2.line(annotated, (ax1, ay1), (ax1 + corner_len, ay1), (0, 220, 255), 3)
            cv2.line(annotated, (ax1, ay1), (ax1, ay1 + corner_len), (0, 220, 255), 3)
            # Top-right
            cv2.line(annotated, (ax2, ay1), (ax2 - corner_len, ay1), (0, 220, 255), 3)
            cv2.line(annotated, (ax2, ay1), (ax2, ay1 + corner_len), (0, 220, 255), 3)
            # Bottom-left
            cv2.line(annotated, (ax1, ay2), (ax1 + corner_len, ay2), (0, 220, 255), 3)
            cv2.line(annotated, (ax1, ay2), (ax1, ay2 - corner_len), (0, 220, 255), 3)
            # Bottom-right
            cv2.line(annotated, (ax2, ay2), (ax2 - corner_len, ay2), (0, 220, 255), 3)
            cv2.line(annotated, (ax2, ay2), (ax2, ay2 - corner_len), (0, 220, 255), 3)

            # Prominent tag banner: "⚠️ ANOMALY LOCATED HERE (85%)"
            tag = f"[!] {a_label} ({score*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(annotated, (ax1, max(0, ay1 - th - 8)), (ax1 + tw + 10, ay1), (0, 120, 230), -1)
            cv2.putText(annotated, tag, (ax1 + 5, max(th, ay1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Top Banner Overlay
        if has_weapon:
            color = (0, 0, 255)
            status_text = "CRITICAL THREAT: WEAPON DETECTED!"
        elif is_anomalous:
            color = (0, 140, 255)
            status_text = "ANOMALY DETECTED: SPATIAL TARGET ACQUIRED"
        else:
            color = (0, 200, 0)
            status_text = "NORMAL ACTIVITY"

        cv2.rectangle(annotated, (0, 0), (w, 65), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            f"Status: {status_text}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            f"Threat Score: {score:.3f} | Thresh: {self.threshold:.2f} | Frame: {frame_idx}",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # Anomaly Score progress bar
        bar_x1, bar_y1 = 15, h - 25
        bar_x2, bar_y2 = w - 15, h - 10
        bar_w = bar_x2 - bar_x1
        
        cv2.rectangle(annotated, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, score)))
        cv2.rectangle(annotated, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), color, -1)
        thresh_x = int(bar_x1 + bar_w * self.threshold)
        cv2.line(annotated, (thresh_x, bar_y1 - 4), (thresh_x, bar_y2 + 4), (255, 255, 255), 2)

        return annotated

    def clear_cache(self):
        """Clear memory cache of processed features and release memory."""
        self._feature_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_video(
        self,
        video_path: str,
        log_to_db: bool = True,
        batch_size: int = 16,
        stride: int = FRAME_STRIDE,
        use_ema: bool = True,
        use_adaptive_threshold: bool = False,
        scan_weapons: bool = True,
        enhance_visibility: bool = True,
        return_annotated: bool = True,
        cache_features: bool = True,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Dual-Stream Video Analysis Pipeline:
        - Normalized ~10 FPS temporal sampling
        - ViT + LSTM Temporal Anomaly Scoring
        - YOLO High-Resolution Weapon Threat Scanning
        - Dynamic DIP Low-Light & Visibility Enhancement (CLAHE + Adaptive Gamma)
        - Sustained anomaly filtering (prevents 1-frame glitches)
        - Thread-safe DB persistence
        """
        cache_key = f"{video_path}_{stride}_{scan_weapons}_{enhance_visibility}"
        
        if progress_callback:
            progress_callback(0.1, "Extracting video frames with OpenCV...")

        if cache_key in self._feature_cache:
            cached_data = self._feature_cache[cache_key]
            frames = cached_data["frames"]
            feature_clips = cached_data["clips_tensor"]
            center_indices = cached_data["center_indices"]
            with torch.inference_mode():
                feature_norm = self.model.feature_norm(feature_clips.to(self.device))
                temporal_rep = self.model.temporal_encoder(feature_norm)
                scores_tensor = self.model.classifier_head(temporal_rep).squeeze(-1).cpu()
        else:
            raw_frames = self.processor.extract_frames_from_video(video_path)
            if not raw_frames:
                return {"error": "Could not read video or empty file"}

            # Auto-detect real surveillance content ROI (strips letterbox/pillarbox/meme captions)
            roi_x1, roi_y1, roi_x2, roi_y2 = self.processor.detect_content_roi(
                sample_frames=raw_frames[:min(len(raw_frames), 30)],
                video_source=video_path
            )
            h_raw, w_raw = raw_frames[0].shape[:2]
            has_borders = (roi_x1 > 0 or roi_y1 > 0 or roi_x2 < w_raw or roi_y2 < h_raw)
            if has_borders and (roi_x2 - roi_x1) >= 64 and (roi_y2 - roi_y1) >= 64:
                frames = [f[roi_y1:roi_y2, roi_x1:roi_x2] for f in raw_frames]
                print(f"[Info] Auto-detected video content ROI: ({roi_x1}, {roi_y1}) -> ({roi_x2}, {roi_y2}) on {w_raw}x{h_raw}")
            else:
                frames = raw_frames

            # Standardize sampling cadence to ~10 FPS (matching ViT+LSTM training: 30fps/3 = 10fps)
            source_fps = self.processor.get_video_fps(video_path)
            sample_step = max(1, round(source_fps / 10.0))
            sampled_frames = frames[::sample_step] if sample_step > 1 else frames

            if progress_callback:
                progress_callback(0.2, f"Preprocessing {len(sampled_frames)} frames (10 FPS normalized)...")

            from src.preprocessing.video_processor import IMAGENET_MEAN, IMAGENET_STD
            valid_rgb = []
            for f in sampled_frames:
                p_f = self.processor.preprocess_frame_fast(
                    f, preserve_aspect=True, enhance_low_light=enhance_visibility
                )
                if p_f is not None:
                    valid_rgb.append(p_f)

            if not valid_rgb:
                return {"error": "No valid frames could be decoded"}

            if len(valid_rgb) < CLIP_LEN:
                valid_rgb = valid_rgb + [valid_rgb[-1]] * (CLIP_LEN - len(valid_rgb))

            np_frames = np.stack(valid_rgb, axis=0)
            t_frames = torch.from_numpy(np_frames).permute(0, 3, 1, 2).float() / 255.0
            t_norm = (t_frames - IMAGENET_MEAN) / IMAGENET_STD

            if progress_callback:
                progress_callback(0.35, f"Extracting {self.spatial_mode.upper()} features ({len(valid_rgb)} frames)...")

            feats_chunks = []
            chunk_size = 32
            total_norm = len(t_norm)
            with torch.inference_mode():
                for b_idx in range(0, total_norm, chunk_size):
                    chunk = t_norm[b_idx : b_idx + chunk_size].to(self.device)
                    feats_chunks.append(self.model.spatial_extractor(chunk).cpu())
                    if progress_callback:
                        progress_callback(
                            0.35 + 0.20 * min(1.0, (b_idx + chunk_size) / max(1, total_norm)),
                            f"Extracting {self.spatial_mode.upper()} features ({min(total_norm, b_idx + chunk_size)}/{total_norm})..."
                        )
            frame_feats = torch.cat(feats_chunks, dim=0)

            clips_list = []
            center_indices = []
            n_f = frame_feats.shape[0]
            for i in range(0, n_f - CLIP_LEN + 1, stride):
                clips_list.append(frame_feats[i : i + CLIP_LEN])
                center_indices.append(min((i + CLIP_LEN // 2) * sample_step, len(frames) - 1))

            if not clips_list:
                clips_list.append(frame_feats[:CLIP_LEN])
                center_indices.append(len(frames) // 2)

            feature_clips = torch.stack(clips_list, dim=0).to(self.device)

            if progress_callback:
                progress_callback(0.6, "Running LSTM temporal sequence analysis...")

            with torch.inference_mode():
                feature_norm = self.model.feature_norm(feature_clips)
                temporal_rep = self.model.temporal_encoder(feature_norm)
                scores_tensor = self.model.classifier_head(temporal_rep).squeeze(-1).cpu()

            if cache_features:
                self._feature_cache[cache_key] = {
                    "frames": frames if return_annotated else [],
                    "clips_tensor": feature_clips,
                    "center_indices": center_indices
                }

        if scores_tensor.dim() == 0:
            raw_scores = [float(scores_tensor.item())]
        else:
            raw_scores = scores_tensor.numpy().tolist()
            if isinstance(raw_scores, float):
                raw_scores = [raw_scores]

        # Calculate Intra-Clip Motion Dynamics (feature velocity per clip)
        clip_motion_energies = []
        with torch.no_grad():
            for c in feature_clips:
                diff = torch.norm(c[1:] - c[:-1], p=2, dim=-1)
                clip_motion_energies.append(float(diff.mean().item()))

        # Calculate effective decision threshold (Static vs Dynamic Adaptive)
        if use_adaptive_threshold and len(raw_scores) > 1:
            mean_s = float(np.mean(raw_scores))
            std_s = float(np.std(raw_scores))
            effective_thresh = max(self.threshold, min(0.95, mean_s + 1.5 * std_s))
        else:
            effective_thresh = self.threshold

        # ── Dual-Stream Threat Scanning: CCTV Weapon & Dangerous Object Detection
        weapon_detected_in_video = False
        weapon_detections_by_clip = {}
        person_detections_by_clip = {}
        vehicle_detections_by_clip = {}
        all_weapon_logs = []

        if self.weapon_model is not None or self.coco_model is not None:
            if progress_callback:
                progress_callback(0.70, "Scanning keyframes for spatial actors, vehicles, and threats...")

            total_c = len(center_indices)
            raw_person_detections = {}
            raw_vehicle_detections = {}

            # Step 1: Detect persons and vehicles across keyframes
            for i, f_idx in enumerate(center_indices):
                raw_k_frame = frames[min(f_idx, len(frames) - 1)]
                t_frame, _ = self.processor.enhance_frame_visibility(raw_k_frame) if enhance_visibility else (raw_k_frame, False)
                f_h, f_w = t_frame.shape[:2] if t_frame is not None and t_frame.ndim >= 2 else (224, 224)
                frame_persons = []
                frame_vehicles = []

                # Primary vehicle & person detection via COCO model if available
                if self.coco_model is not None:
                    try:
                        c_res = self.coco_model(t_frame, verbose=False, conf=0.16)
                        if c_res and len(c_res) > 0 and c_res[0].boxes is not None:
                            for box in c_res[0].boxes:
                                cls_id = int(box.cls[0].item())
                                cls_name = self.coco_model.names.get(cls_id, "").lower()
                                conf = float(box.conf[0].item())
                                xyxy = [int(v) for v in box.xyxy[0].cpu().numpy().tolist()]

                                bw = xyxy[2] - xyxy[0]
                                bh = xyxy[3] - xyxy[1]

                                if cls_name == "person" and conf >= 0.22:
                                    if bw >= 0.04 * f_w and bh >= 0.08 * f_h:
                                        frame_persons.append(xyxy)
                                elif any(v in cls_name for v in ["car", "motorcycle", "bus", "truck", "bicycle"]):
                                    frame_vehicles.append(xyxy)
                    except Exception:
                        pass

                # Fallback person detection via weapon model if COCO didn't find persons
                if self.weapon_model is not None and not frame_persons:
                    try:
                        w_res = self.weapon_model(t_frame, verbose=False, conf=0.22)
                        if w_res and len(w_res) > 0 and w_res[0].boxes is not None:
                            for box in w_res[0].boxes:
                                cls_id = int(box.cls[0].item())
                                cls_name = self.weapon_model.names.get(cls_id, "").lower()
                                conf = float(box.conf[0].item())
                                xyxy = [int(v) for v in box.xyxy[0].cpu().numpy().tolist()]
                                bw = xyxy[2] - xyxy[0]
                                bh = xyxy[3] - xyxy[1]
                                if cls_name == "person" and conf >= 0.25:
                                    if bw >= 0.04 * f_w and bh >= 0.08 * f_h:
                                        frame_persons.append(xyxy)
                    except Exception:
                        pass

                raw_person_detections[i] = frame_persons
                raw_vehicle_detections[i] = frame_vehicles

            # Step 2: Bidirectional Temporal Box Propagation for Persons & Vehicles
            for i in range(total_c):
                p_list = raw_person_detections.get(i, [])
                if not p_list:
                    for step in [1, -1, 2, -2]:
                        neighbor = i + step
                        if 0 <= neighbor < total_c and raw_person_detections.get(neighbor):
                            p_list = [list(box) for box in raw_person_detections[neighbor]]
                            break
                person_detections_by_clip[i] = p_list

                v_list = raw_vehicle_detections.get(i, [])
                if not v_list:
                    for step in [1, -1, 2, -2, 3, -3]:
                        neighbor = i + step
                        if 0 <= neighbor < total_c and raw_vehicle_detections.get(neighbor):
                            v_list = [list(box) for box in raw_vehicle_detections[neighbor]]
                            break
                vehicle_detections_by_clip[i] = v_list

            # Step 3: Verified Weapon & Threat Scanning
            # Enforces:
            #  1) Vehicle Veto (cars, motorcycles, bicycles rejected)
            #  2) Body & Anatomical Hand Veto (bare hands, arms, whole bodies rejected)
            #  3) Scale Sanity (weapons are handheld tools, not vehicle sized)
            #  4) Dynamic Motion Verification (stationary rails/fixtures rejected)
            #  5) Human Proximity Verification (must be handheld or brandished by actor)
            for i, f_idx in enumerate(center_indices):
                raw_k_frame = frames[min(f_idx, len(frames) - 1)]
                t_frame, _ = self.processor.enhance_frame_visibility(raw_k_frame) if enhance_visibility else (raw_k_frame, False)
                f_h, f_w = t_frame.shape[:2] if t_frame is not None and t_frame.ndim >= 2 else (224, 224)
                frame_weapons = []

                # Compute motion difference against adjacent reference frame
                ref_idx = max(0, f_idx - 2) if f_idx >= 2 else min(len(frames) - 1, f_idx + 2)
                r_frame = frames[ref_idx] if ref_idx != f_idx else None
                diff = None
                global_motion = 0.0
                if r_frame is not None and r_frame.shape == t_frame.shape:
                    try:
                        g_curr = cv2.cvtColor(t_frame, cv2.COLOR_BGR2GRAY)
                        g_ref = cv2.cvtColor(r_frame, cv2.COLOR_BGR2GRAY)
                        diff = cv2.absdiff(g_curr, g_ref)
                        global_motion = float(np.mean(diff))
                    except Exception:
                        diff = None

                p_candidates = person_detections_by_clip.get(i, [])
                v_candidates = vehicle_detections_by_clip.get(i, [])

                try:
                    w_res = self.weapon_model(t_frame, verbose=False, conf=0.45) if self.weapon_model else None
                    if w_res and len(w_res) > 0 and w_res[0].boxes is not None:
                        for box in w_res[0].boxes:
                            cls_id = int(box.cls[0].item())
                            cls_name = self.weapon_model.names.get(cls_id, "").lower()
                            conf = float(box.conf[0].item())
                            xyxy = box.xyxy[0].cpu().numpy().tolist()

                            bw = xyxy[2] - xyxy[0]
                            bh = xyxy[3] - xyxy[1]
                            area = bw * bh
                            aspect = max(bw / max(1e-3, bh), bh / max(1e-3, bw))

                            # 1. SCALE SANITY: Handheld weapons (guns, knives, bats) are compact instruments
                            # Rejects car/motorcycle or room-scale bounding boxes
                            is_scale_valid = (
                                bw <= 0.28 * f_w and
                                bh <= 0.35 * f_h and
                                area <= 0.065 * (f_w * f_h) and
                                area >= 200 and
                                aspect <= 4.5
                            )

                            is_threat_class = any(
                                w in cls_name for w in ["weapon", "knife", "gun", "bat", "scissor", "pistol", "rifle", "blade", "dagger"]
                            )

                            if not (scan_weapons and conf >= 0.50 and is_threat_class and is_scale_valid):
                                continue

                            # 2. VEHICLE VETO: Rejects cars, motorcycles, bicycles, buses, trucks
                            if self._is_vehicle_overlap(xyxy, v_candidates):
                                continue

                            # 3. WHOLE-BODY PERSON VETO: Rejects human bodies misclassified as weapons
                            if self._is_person_body_overlap(xyxy, p_candidates):
                                continue

                            # 4. ANATOMICAL HAND & LIMB VETO: Rejects bare hands, fists, arms, gestures
                            w_crop = t_frame[max(0, int(xyxy[1])):min(f_h, int(xyxy[3])), max(0, int(xyxy[0])):min(f_w, int(xyxy[2]))]
                            if self._is_human_hand_or_limb(w_crop):
                                continue

                            # 5. CEILING & OVERHEAD FIXTURE REJECTION:
                            if (xyxy[1] <= 0.06 * f_h) and (xyxy[3] <= 0.22 * f_h):
                                continue

                            # 6. DYNAMIC MOTION VERIFICATION:
                            # Real weapons in surveillance altercations move with the combatants.
                            is_moving = False
                            if diff is not None:
                                wx1, wy1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
                                wx2, wy2 = min(f_w, int(xyxy[2])), min(f_h, int(xyxy[3]))
                                if wx2 > wx1 and wy2 > wy1:
                                    w_crop_diff = diff[wy1:wy2, wx1:wx2]
                                    box_motion = float(np.mean(w_crop_diff))
                                    if box_motion >= 3.5 and (global_motion < 1.0 or box_motion >= 1.25 * global_motion):
                                        is_moving = True
                            else:
                                is_moving = True  # fallback if diff unavailable

                            if not is_moving:
                                continue  # Reject stationary metal fixture / pole

                            # 7. HANDHELD / HUMAN ACTOR PROXIMITY VERIFICATION:
                            # A weapon must be in hand or within reaching proximity of a human actor.
                            is_near_actor = False
                            if p_candidates:
                                max_reach = 0.28 * max(f_w, f_h)
                                for p in p_candidates:
                                    if self._box_distance(xyxy, p) <= max_reach:
                                        is_near_actor = True
                                        break
                            else:
                                is_near_actor = (conf >= 0.72)

                            if not is_near_actor:
                                continue  # Reject isolated fixture far from any human actor

                            # Confirmed dynamic threat
                            frame_weapons.append({
                                "box": xyxy,
                                "label": cls_name,
                                "confidence": conf
                            })
                            all_weapon_logs.append({
                                "frame": f_idx,
                                "clip_idx": i,
                                "type": cls_name,
                                "confidence": conf
                            })

                    if frame_weapons:
                        weapon_detections_by_clip[i] = frame_weapons
                except Exception:
                    pass

        has_confirmed_weapon = len(weapon_detections_by_clip) > 0
        weapon_detected_in_video = has_confirmed_weapon

        # ── Dynamic Temporal Motion & Threat Calibration for Clip-Level Discrimination
        # Resolves static scene saturation (where LSTM outputs ~0.999 across all clips of a video)
        # Disentangles normal quiescent segments from acute anomaly intervals
        raw_min = float(np.min(raw_scores)) if raw_scores else 0.0
        raw_max = float(np.max(raw_scores)) if raw_scores else 0.0
        is_saturated = (raw_min > 0.70) and ((raw_max - raw_min) < 0.15)

        m_np = np.array(clip_motion_energies) if clip_motion_energies else np.zeros(len(raw_scores))
        m_min = float(np.min(m_np)) if len(m_np) > 0 else 0.0
        m_max = float(np.max(m_np)) if len(m_np) > 0 else 0.0
        m_base = float(np.percentile(m_np, 20)) if len(m_np) > 0 else 0.0
        m_range = max(1e-5, m_max - m_base)

        calibrated_scores = []
        for i, r_score in enumerate(raw_scores):
            m_val = clip_motion_energies[i] if i < len(clip_motion_energies) else 0.0
            has_w = i in weapon_detections_by_clip
            w_conf = max(w["confidence"] for w in weapon_detections_by_clip[i]) if has_w else 0.0

            if has_w:
                s = max(r_score, 0.90, w_conf)
            elif is_saturated and m_range > 1.2:
                # Dynamic intra-video motion modulation:
                # Accurately maps calm baseline periods to normal (< 0.35) and active disruption to anomaly (> 0.35)
                rel = float(np.clip((m_val - m_base) / m_range, 0.0, 1.0))
                if rel < 0.28:
                    # Calm/normal resting baseline interval -> Below threshold (0.12 - 0.32)
                    s = 0.12 + (rel / 0.28) * 0.20
                else:
                    # Active anomaly / struggle / disruption -> Above threshold (0.38 - 0.96)
                    s = 0.38 + ((rel - 0.28) / (1.0 - 0.28)) * 0.58
            else:
                s = float(r_score)

            calibrated_scores.append(float(s))

        # Apply EMA smoothing for temporal continuity
        scores = self._apply_ema_smoothing(calibrated_scores) if use_ema else calibrated_scores

        # Ensure confirmed weapon detections never get damped below threshold by EMA
        for i in range(len(scores)):
            if i in weapon_detections_by_clip:
                w_conf = max(w["confidence"] for w in weapon_detections_by_clip[i])
                scores[i] = max(scores[i], 0.88, w_conf)

        is_anomalous_list = []
        anomaly_rois_by_clip = {}
        annotated_frames = []
        total_s = len(scores)

        for i, score in enumerate(scores):
            if progress_callback and (i % 5 == 0 or i == total_s - 1):
                progress_callback(0.85 + 0.15 * ((i + 1) / max(1, total_s)), f"Rendering telemetry overlays ({i + 1}/{total_s})...")
            
            anom_flag = score >= effective_thresh
            is_anomalous_list.append(anom_flag)

            # Localize spatial anomaly ROI for any flagged anomalous clip
            curr_roi = None
            w_boxes = weapon_detections_by_clip.get(i, None)
            if anom_flag:
                f_curr_idx = center_indices[i]
                target_frame = frames[min(f_curr_idx, len(frames) - 1)]
                f_ref_idx = max(0, f_curr_idx - 3) if f_curr_idx >= 3 else min(len(frames) - 1, f_curr_idx + 3)
                ref_frame = frames[f_ref_idx] if f_ref_idx != f_curr_idx else None
                p_boxes = person_detections_by_clip.get(i, None)
                curr_roi = self.localize_anomaly_region(
                    target_frame, ref_frame, person_boxes=p_boxes, weapon_boxes=w_boxes
                )
                anomaly_rois_by_clip[i] = curr_roi

            if return_annotated:
                frame_idx = center_indices[i]
                target_frame = frames[min(frame_idx, len(frames) - 1)]
                ann_frame = self.draw_annotation(
                    target_frame, score, anom_flag, frame_idx,
                    weapon_boxes=w_boxes,
                    anomaly_roi=curr_roi
                )
                annotated_frames.append(ann_frame)

        # ── Compute final result metrics
        final_max   = max(scores) if scores else 0.0
        final_avg   = float(np.mean(scores)) if scores else 0.0
        anom_clips  = sum(is_anomalous_list)

        # ── Robust Sustained Anomaly Filtering (Eliminate 1-frame glitches on normal videos)
        anom_ratio = anom_clips / max(1, total_s)
        has_consecutive = any(
            is_anomalous_list[k] and is_anomalous_list[k + 1]
            for k in range(len(is_anomalous_list) - 1)
        ) if total_s > 1 else (is_anomalous_list[0] if total_s == 1 else False)

        video_anomaly_detected = bool(
            weapon_detected_in_video or
            anom_ratio >= 0.04 or
            (has_consecutive and final_max >= effective_thresh) or
            (total_s <= 2 and final_max >= effective_thresh)
        )

        anomaly_classification = (
            "Weapon Threat Detected" if weapon_detected_in_video
            else ("Abnormal / Violent Motion" if video_anomaly_detected else "Normal Activity")
        )

        # ── Persist to database (3-table schema)
        session_id = None
        if log_to_db:
            try:
                # 1. Log full session summary
                session_id = self.db.log_session(
                    video_source=Path(video_path).name,
                    spatial_mode=self.spatial_mode,
                    threshold=effective_thresh,
                    total_frames=len(frames),
                    total_clips=len(scores),
                    anomaly_clips=anom_clips,
                    max_score=final_max,
                    avg_score=final_avg,
                    anomaly_detected=video_anomaly_detected,
                    use_ema=use_ema,
                    use_adaptive=use_adaptive_threshold,
                    notes=f"type={anomaly_classification}"
                )

                # 2. Bulk-insert full clip-by-clip score timeseries
                self.db.log_score_timeseries(
                    session_id=session_id,
                    raw_scores=raw_scores,
                    ema_scores=scores,
                    is_anomalous=is_anomalous_list
                )

                # 3. Log individual anomalous-clip events (deduplicated by frame proximity)
                last_logged_frame = -100
                for i, (score, anom_flag) in enumerate(zip(scores, is_anomalous_list)):
                    frame_idx = center_indices[i]
                    if anom_flag and (frame_idx - last_logged_frame > 15):
                        self.db.log_event(
                            video_source=Path(video_path).name,
                            frame_number=frame_idx,
                            anomaly_score=score,
                            classification=anomaly_classification,
                            threshold=effective_thresh,
                            notes=f"Score {score:.4f} exceeded threshold {effective_thresh:.2f}",
                            session_id=session_id
                        )
                        last_logged_frame = frame_idx

            except Exception as e:
                print(f"[Warning] DB logging failed: {e}")

        return {
            "total_frames":        len(frames),
            "total_clips":         len(scores),
            "anomaly_clips":       anom_clips,
            "raw_scores":          raw_scores,
            "scores":              scores,
            "center_indices":      center_indices,
            "is_anomalous":        is_anomalous_list,
            "anomaly_rois":        anomaly_rois_by_clip,
            "annotated_frames":    annotated_frames,
            "max_score":           final_max,
            "avg_score":           final_avg,
            "anomaly_detected":    video_anomaly_detected,
            "anomaly_type":        anomaly_classification,
            "weapon_detected":     weapon_detected_in_video,
            "weapon_detections":   all_weapon_logs,
            "session_id":          session_id,
        }

if __name__ == "__main__":
    detector = VideoAnomalyDetector()
    print("[Self-Test] VideoAnomalyDetector with EMA smoothing initialized successfully!")
