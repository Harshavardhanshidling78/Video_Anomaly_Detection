import cv2
import numpy as np
import torch
from typing import List, Tuple, Generator, Optional
from src.config import FRAME_WIDTH, FRAME_HEIGHT, CLIP_LEN, FRAME_STRIDE

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

class FastVideoProcessor:
    """
    Production-Grade Vectorized Video Preprocessor with:
    - OOM (Out Of Memory) Dynamic Allocation Safeguards
    - Corrupted / Bad Frame Auto-Recovery
    - Live RTSP Camera & HTTP Video Stream Ingestion
    """
    def __init__(self, target_size: Tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT)):
        self.target_size = target_size
        cv2.setNumThreads(cv2.getNumberOfCPUs())

    @staticmethod
    def enhance_frame_visibility(
        frame_bgr: np.ndarray,
        brightness_thresh: float = 70.0,
        contrast_thresh: float = 25.0
    ) -> Tuple[np.ndarray, bool]:
        """
        Intelligent Digital Image Processing (DIP) Visibility Enhancer:
        - If the frame is ALREADY properly visible (well-lit, good contrast), it is left 100% untouched.
        - If the frame has poor visibility (dark, night, dim shadows, or washed-out flat contrast):
          Applies adaptive Luminance-channel CLAHE + Adaptive Gamma correction + Bilateral edge-preserving denoise.
        Returns: (enhanced_or_original_frame, was_enhanced_boolean)
        """
        if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
            return frame_bgr, False

        try:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            mean_l = float(np.mean(l_channel))
            std_l = float(np.std(l_channel))

            # Condition: Frame is already properly visible
            if mean_l >= brightness_thresh and std_l >= contrast_thresh:
                return frame_bgr, False

            # Frame suffers from low visibility / darkness / poor contrast -> Apply targeted DIP
            enhanced_l = l_channel.copy()

            # 1. Adaptive Non-linear Gamma Correction for Deep Shadows (gamma < 1.0 brightens shadows)
            if mean_l < 60.0:
                gamma = max(0.50, min(0.85, 0.50 + 0.35 * (mean_l / 60.0)))
                table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
                enhanced_l = cv2.LUT(enhanced_l, table)

            # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(enhanced_l)

            # 3. Mild Edge-Preserving Bilateral Denoising if severe low light
            if mean_l < 50.0:
                enhanced_l = cv2.bilateralFilter(enhanced_l, d=5, sigmaColor=30, sigmaSpace=30)

            merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
            return enhanced_bgr, True
        except Exception:
            return frame_bgr, False

    def preprocess_frame_fast(
        self,
        frame_bgr: np.ndarray,
        preserve_aspect: bool = True,
        enhance_low_light: bool = True
    ) -> Optional[np.ndarray]:
        """
        Fast C++ OpenCV resize + RGB conversion with aspect-ratio preserving letterbox padding.
        Optionally enhances dark / low-visibility frames with adaptive DIP CLAHE + Gamma.
        Prevents geometrical distortion on vertical phone or widescreen surveillance feeds.
        """
        if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
            return None
        try:
            # Apply low-visibility DIP enhancement only if the frame needs it
            if enhance_low_light:
                working_frame, _ = self.enhance_frame_visibility(frame_bgr)
            else:
                working_frame = frame_bgr

            if not preserve_aspect:
                resized_bgr = cv2.resize(working_frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            else:
                h, w = working_frame.shape[:2]
                target_w, target_h = self.target_size
                scale = min(target_w / w, target_h / h)
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                scaled = cv2.resize(working_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                resized_bgr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                dx = (target_w - new_w) // 2
                dy = (target_h - new_h) // 2
                resized_bgr[dy : dy + new_h, dx : dx + new_w] = scaled

            return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def get_video_fps(self, video_source: str) -> float:
        """Returns detected video FPS with safe fallback."""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            return 30.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 0 else 30.0

    def detect_content_roi(
        self,
        sample_frames: Optional[List[np.ndarray]] = None,
        video_source: Optional[str] = None
    ) -> Tuple[int, int, int, int]:
        """
        Detects the real CCTV surveillance content region (bounding box x1, y1, x2, y2)
        by stripping letterboxing, pillarboxing, meme captions, or watermarked borders.
        Uses ultra-fast, zero-memory downscaled accumulator.
        Returns: (x1, y1, x2, y2)
        """
        pw, ph = 120, 160
        accum = np.zeros((ph, pw), dtype=np.float32)
        count = 0
        orig_w, orig_h = 0, 0

        if sample_frames and len(sample_frames) > 0:
            orig_h, orig_w = sample_frames[0].shape[:2]
            step = max(1, len(sample_frames) // 8)
            for f in sample_frames[::step]:
                if f is None or f.size == 0:
                    continue
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                small = cv2.resize(g, (pw, ph), interpolation=cv2.INTER_AREA)
                accum += small
                count += 1
                if count >= 8:
                    break
        elif video_source:
            cap = cv2.VideoCapture(video_source)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                step = max(1, total // 8) if total > 0 else 1
                for idx in range(0, max(1, total), step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    small = cv2.resize(g, (pw, ph), interpolation=cv2.INTER_AREA)
                    accum += small
                    count += 1
                    if count >= 8:
                        break
                cap.release()

        if count == 0 or orig_w <= 0 or orig_h <= 0:
            return 0, 0, max(1, orig_w), max(1, orig_h)

        avg = accum / float(count)
        row_stds = np.std(avg, axis=1)
        row_means = np.mean(avg, axis=1)
        row_active = (row_stds > 10) & (row_means > 15)

        def find_block(mask, total_len, min_border_pct=0.06):
            blocks = []
            start = None
            for i, val in enumerate(mask):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    blocks.append((start, i))
                    start = None
            if start is not None:
                blocks.append((start, total_len))
            if not blocks:
                return 0, total_len
            blocks.sort(key=lambda b: b[1] - b[0], reverse=True)
            b_start, b_end = blocks[0]
            if (b_end - b_start) < 0.20 * total_len:
                return 0, total_len
            # Only crop if borders are actually present (> min_border_pct)
            start_crop = b_start if b_start >= int(min_border_pct * total_len) else 0
            end_crop = b_end if (total_len - b_end) >= int(min_border_pct * total_len) else total_len
            return start_crop, end_crop

        py1, py2 = find_block(row_active, ph)
        band = avg[py1:py2, :]
        col_stds = np.std(band, axis=0)
        col_means = np.mean(band, axis=0)
        col_active = (col_stds > 8) & (col_means > 12)
        px1, px2 = find_block(col_active, pw)

        sy = orig_h / float(ph)
        sx = orig_w / float(pw)
        x1 = max(0, int(round(px1 * sx)))
        x2 = min(orig_w, int(round(px2 * sx)))
        y1 = max(0, int(round(py1 * sy)))
        y2 = min(orig_h, int(round(py2 * sy)))

        return x1, y1, x2, y2

    def extract_frames_from_video(
        self, video_source: str, max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extracts frames safely from Local Files, RTSP streams, or HTTP URLs.
        Includes corrupted frame auto-skipping.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[Warning] Could not open video stream source: {video_source}")
            return []

        frames = []
        count = 0
        consecutive_failures = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    break
                continue
            
            consecutive_failures = 0
            frames.append(frame)
            count += 1
            if max_frames and count >= max_frames:
                break

        cap.release()
        return frames

    def create_clips_batch(
        self,
        frames: List[np.ndarray],
        clip_len: int = CLIP_LEN,
        stride: int = FRAME_STRIDE,
        max_chunk_clips: int = 64
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Vectorized clip builder creating 5D Tensors with OOM memory chunk protection.
        """
        valid_rgb = []
        for f in frames:
            p_f = self.preprocess_frame_fast(f)
            if p_f is not None:
                valid_rgb.append(p_f)

        n_frames = len(valid_rgb)
        if n_frames == 0:
            return torch.empty(0, clip_len, 3, self.target_size[1], self.target_size[0]), []

        if n_frames < clip_len:
            padded = valid_rgb + [valid_rgb[-1]] * (clip_len - n_frames)
            np_clip = np.stack(padded, axis=0) # (T, H, W, 3)
            clip_t = torch.from_numpy(np_clip).permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
            clip_t = ((clip_t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)
            return clip_t, [n_frames // 2]

        clip_list = []
        center_indices = []

        for i in range(0, n_frames - clip_len + 1, stride):
            sub_rgb = valid_rgb[i : i + clip_len]
            clip_list.append(sub_rgb)
            center_indices.append(i + clip_len // 2)

        # OOM Safeguard: Chunk tensor creation if clip list is huge
        all_tensor_chunks = []
        total_clips = len(clip_list)

        for chunk_idx in range(0, total_clips, max_chunk_clips):
            sub_clips = clip_list[chunk_idx : chunk_idx + max_chunk_clips]
            np_chunk = np.stack(sub_clips, axis=0) # (Chunk, T, H, W, 3)
            t_chunk = torch.from_numpy(np_chunk).permute(0, 1, 4, 2, 3).to(dtype=torch.float32) / 255.0
            t_norm = (t_chunk - IMAGENET_MEAN.unsqueeze(0)) / IMAGENET_STD.unsqueeze(0)
            all_tensor_chunks.append(t_norm)

        normalized_all = torch.cat(all_tensor_chunks, dim=0)
        return normalized_all, center_indices

# Alias for compatibility
VideoProcessor = FastVideoProcessor
# Member 2 Video Processing Pipeline Verified by Shreyans