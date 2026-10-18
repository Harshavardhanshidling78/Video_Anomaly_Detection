import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Generator, Tuple, Optional
from src.config import DEVICE, ANOMALY_THRESHOLD, CHECKPOINT_DIR, CLIP_LEN, FRAME_STRIDE, RECURRENT_TYPE

from src.preprocessing.video_processor import FastVideoProcessor
from src.models.anomaly_classifier import VideoAnomalyDetectorModel
from src.utils.database import AnomalyDatabase

class VideoAnomalyDetector:
    """
    Enterprise Video Anomaly Detector Pipeline with:
    - C++ OpenCV fast frame extraction & RTSP stream ingestion
    - EMA (Exponential Moving Average) temporal noise smoothing
    - Dynamic adaptive thresholding (mean + k * std)
    - Feature caching for instant re-evaluations (< 0.01s)
    - Thread-safe SQLite incident auditing
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        spatial_mode: str = "fast",
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
        
        # Initialize Deep Learning Model (ViT / MobileNet + LSTM/GRU)
        self.model = VideoAnomalyDetectorModel(spatial_mode=spatial_mode, threshold=threshold).to(self.device)
        self.model.eval()

        if model_path and Path(model_path).exists():
            try:
                checkpoint_obj = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
                    ckpt_mode = checkpoint_obj.get("spatial_mode", spatial_mode)
                    ckpt_cell = checkpoint_obj.get("cell_type", RECURRENT_TYPE)
                    if ckpt_mode != self.spatial_mode or ckpt_cell != self.model.temporal_encoder.cell_type:
                        self.spatial_mode = ckpt_mode
                        self.model = VideoAnomalyDetectorModel(spatial_mode=ckpt_mode, cell_type=ckpt_cell, threshold=threshold).to(self.device)
                    self.model.load_state_dict(checkpoint_obj["state_dict"])
                else:
                    self.model.load_state_dict(checkpoint_obj)
                self.model.eval()
                print(f"[Info] Loaded trained model checkpoint from {model_path}")
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint ({e}). Using baseline initialized weights.")


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

    def draw_annotation(
        self,
        frame: np.ndarray,
        score: float,
        is_anomalous: bool,
        frame_idx: int,
        max_display_width: int = 640
    ) -> np.ndarray:
        """
        Fast visual banner annotation onto video frame (downsampled for low memory usage).
        """
        if frame is None or frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        if w > max_display_width:
            new_h = int(h * (max_display_width / w))
            annotated = cv2.resize(frame, (max_display_width, new_h), interpolation=cv2.INTER_AREA)
            h, w = annotated.shape[:2]
        else:
            annotated = frame.copy()

        color = (0, 0, 255) if is_anomalous else (0, 200, 0)
        status_text = "ANOMALY DETECTED!" if is_anomalous else "NORMAL ACTIVITY"

        # Top overlay banner
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
            f"Anomaly Score: {score:.3f} | Thresh: {self.threshold:.2f} | Frame: {frame_idx}",
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

    def process_video(
        self,
        video_path: str,
        log_to_db: bool = True,
        batch_size: int = 16,
        stride: int = FRAME_STRIDE,
        use_ema: bool = True,
        use_adaptive_threshold: bool = False,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Production Video Analysis Pipeline with EMA filtering, adaptive thresholding, & progress reporting.
        """
        cache_key = f"{video_path}_{stride}"
        
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

            sample_step = 2 if len(raw_frames) > 50 else 1
            sampled_frames = raw_frames[::sample_step] if sample_step > 1 else raw_frames

            if progress_callback:
                progress_callback(0.2, f"Preprocessing {len(sampled_frames)} frames...")

            from src.preprocessing.video_processor import IMAGENET_MEAN, IMAGENET_STD
            valid_rgb = []
            for f in sampled_frames:
                p_f = self.processor.preprocess_frame_fast(f)
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
                progress_callback(0.4, f"Extracting {self.spatial_mode.upper()} features ({len(valid_rgb)} frames)...")

            with torch.inference_mode():
                frame_feats = self.model.spatial_extractor(t_norm.to(self.device)).cpu()

            clips_list = []
            center_indices = []
            n_f = frame_feats.shape[0]
            for i in range(0, n_f - CLIP_LEN + 1, stride):
                clips_list.append(frame_feats[i : i + CLIP_LEN])
                center_indices.append((i + CLIP_LEN // 2) * sample_step)

            if not clips_list:
                clips_list.append(frame_feats[:CLIP_LEN])
                center_indices.append(len(raw_frames) // 2)

            feature_clips = torch.stack(clips_list, dim=0).to(self.device)

            if progress_callback:
                progress_callback(0.7, "Running LSTM temporal sequence analysis...")

            with torch.inference_mode():
                feature_norm = self.model.feature_norm(feature_clips)
                temporal_rep = self.model.temporal_encoder(feature_norm)
                scores_tensor = self.model.classifier_head(temporal_rep).squeeze(-1).cpu()

            self._feature_cache[cache_key] = {
                "frames": raw_frames,
                "clips_tensor": feature_clips,
                "center_indices": center_indices
            }
            frames = raw_frames

        if scores_tensor.dim() == 0:
            raw_scores = [float(scores_tensor.item())]
        else:
            raw_scores = scores_tensor.numpy().tolist()
            if isinstance(raw_scores, float):
                raw_scores = [raw_scores]

        # Apply Exponential Moving Average (EMA) smoothing
        scores = self._apply_ema_smoothing(raw_scores) if use_ema else raw_scores

        # Calculate effective decision threshold (Static vs Dynamic Adaptive)
        if use_adaptive_threshold and len(scores) > 1:
            mean_s = float(np.mean(scores))
            std_s = float(np.std(scores))
            effective_thresh = max(self.threshold, min(0.95, mean_s + 1.5 * std_s))
        else:
            effective_thresh = self.threshold

        is_anomalous_list = []
        annotated_frames = []
        last_logged_frame = -100
        total_s = len(scores)

        for i, score in enumerate(scores):
            if progress_callback and (i % 5 == 0 or i == total_s - 1):
                progress_callback(0.7 + 0.3 * ((i + 1) / max(1, total_s)), f"Rendering telemetry overlays ({i + 1}/{total_s})...")
            anom_flag = score >= effective_thresh
            is_anomalous_list.append(anom_flag)

            frame_idx = center_indices[i]
            target_frame = frames[min(frame_idx, len(frames) - 1)]
            
            ann_frame = self.draw_annotation(target_frame, score, anom_flag, frame_idx)
            annotated_frames.append(ann_frame)

            if log_to_db and anom_flag and (frame_idx - last_logged_frame > 15):
                self.db.log_event(
                    video_source=Path(video_path).name,
                    frame_number=frame_idx,
                    anomaly_score=score,
                    classification="Anomalous",
                    threshold=effective_thresh,
                    notes=f"Exceeded alert boundary {effective_thresh:.2f}"
                )
                last_logged_frame = frame_idx


        return {
            "total_frames": len(frames),
            "total_clips": len(scores),
            "raw_scores": raw_scores,
            "scores": scores,
            "center_indices": center_indices,
            "is_anomalous": is_anomalous_list,
            "annotated_frames": annotated_frames,
            "max_score": max(scores) if scores else 0.0,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "anomaly_detected": any(is_anomalous_list)
        }

if __name__ == "__main__":
    detector = VideoAnomalyDetector()
    print("[Self-Test] VideoAnomalyDetector with EMA smoothing initialized successfully!")
