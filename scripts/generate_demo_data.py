import sys
import cv2
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DEMO_DIR

def create_demo_videos():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    fps = 20
    duration_sec = 6 # 120 frames total

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 1. Normal Video Creation: Smooth linear horizontal motion across scene
    normal_path = DEMO_DIR / "normal_surveillance_demo.mp4"
    out_normal = cv2.VideoWriter(str(normal_path), fourcc, fps, (width, height))

    for frame_idx in range(duration_sec * fps):
        # Dark background (simulating hallway/parking lot)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (35, 35, 40)
        
        # Grid lines (background structure)
        cv2.line(img, (0, 350), (width, 350), (80, 80, 80), 2)
        
        # Person/object moving smoothly from left to right
        cx = int(50 + (frame_idx / (duration_sec * fps)) * (width - 100))
        cy = 300
        
        # Draw walking person silhouette representation
        cv2.circle(img, (cx, cy - 40), 15, (0, 200, 100), -1) # Head
        cv2.rectangle(img, (cx - 10, cy - 25), (cx + 10, cy + 30), (0, 200, 100), -1) # Body
        
        # Subdued text
        cv2.putText(img, "CAM 01 - MAIN CORRIDOR (NORMAL)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        out_normal.write(img)

    out_normal.release()
    print(f"[Success] Generated normal video clip: {normal_path}")

    # 2. Anomalous Video Creation: Starts normal, then undergoes sudden chaotic motion / fall / fight disturbance
    anom_path = DEMO_DIR / "anomalous_event_demo.mp4"
    out_anom = cv2.VideoWriter(str(anom_path), fourcc, fps, (width, height))

    for frame_idx in range(duration_sec * fps):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (35, 35, 40)
        cv2.line(img, (0, 350), (width, 350), (80, 80, 80), 2)
        
        cv2.putText(img, "CAM 02 - PARKING BAY (SURVEILLANCE)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if frame_idx < 40:
            # Normal segment (first 2 seconds)
            cx = int(80 + frame_idx * 4)
            cy = 300
            cv2.circle(img, (cx, cy - 40), 15, (220, 180, 50), -1)
            cv2.rectangle(img, (cx - 10, cy - 25), (cx + 10, cy + 30), (220, 180, 50), -1)
        else:
            # Anomalous disturbance segment (Frames 40 to 120): erratic sudden movements & rapid flashing
            noise_offset_x = np.random.randint(-40, 40)
            noise_offset_y = np.random.randint(-30, 30)
            cx = int(240 + noise_offset_x)
            cy = int(300 + noise_offset_y)

            # Rapid chaotic interaction / disturbance representation
            cv2.circle(img, (cx, cy - 20), 20, (50, 50, 240), -1)
            cv2.rectangle(img, (cx - 25, cy - 10), (cx + 25, cy + 40), (30, 30, 220), -1)
            
            # Secondary erratic entity (fight / rapid interaction)
            cx2 = int(240 - noise_offset_x)
            cy2 = int(300 - noise_offset_y)
            cv2.circle(img, (cx2, cy2 - 20), 20, (0, 140, 255), -1)
            cv2.rectangle(img, (cx2 - 20, cy2 - 10), (cx2 + 20, cy2 + 40), (0, 140, 255), -1)
            
            # Flash disturbance lines
            cv2.line(img, (cx - 30, cy), (cx + 30, cy), (255, 255, 255), 2)
            cv2.line(img, (cx, cy - 30), (cx, cy + 30), (255, 255, 255), 2)

        out_anom.write(img)

    out_anom.release()
    print(f"[Success] Generated anomalous video clip: {anom_path}")

if __name__ == "__main__":
    create_demo_videos()
