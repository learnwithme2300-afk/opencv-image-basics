from pathlib import Path
import cv2

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_image(out_dir: str, name: str, img):
    out_path = ensure_dir(out_dir) / name
    cv2.imwrite(str(out_path), img)
    return out_path
