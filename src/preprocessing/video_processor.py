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

    def preprocess_frame_fast(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Fast C++ OpenCV resize + RGB conversion with corrupted frame validation.
        """
        if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
            return None
        try:
            resized_bgr = cv2.resize(frame_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)
            return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

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