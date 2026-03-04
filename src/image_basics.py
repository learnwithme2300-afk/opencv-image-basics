import argparse
from pathlib import Path
import cv2
import numpy as np
from utils import save_image, ensure_dir

def main():
    parser = argparse.ArgumentParser(description="OpenCV image basics demo")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="outputs", help="Output folder")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    out_dir = ensure_dir(args.out)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError("Failed to read image (format unsupported or corrupted).")

    # 1) Original
    save_image(str(out_dir), "01_original.jpg", img_bgr)

    # 2) Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    save_image(str(out_dir), "02_gray.jpg", gray)

    # 3) Blur (Gaussian)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    save_image(str(out_dir), "03_blur.jpg", blur)

    # 4) Histogram equalization
    eq = cv2.equalizeHist(gray)
    save_image(str(out_dir), "04_equalized.jpg", eq)

    # 5) Threshold (binary + adaptive)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_image(str(out_dir), "05_threshold_otsu.jpg", th)

    ad = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    save_image(str(out_dir), "06_threshold_adaptive.jpg", ad)

    # 6) Edges (Sobel + Canny)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = np.uint8(np.clip(mag / (mag.max() + 1e-9) * 255, 0, 255))
    save_image(str(out_dir), "07_edges_sobel_mag.jpg", mag)

    canny = cv2.Canny(gray, 80, 160)
    save_image(str(out_dir), "08_edges_canny.jpg", canny)

    # 7) Morphology (erode/dilate)
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(th, kernel, iterations=1)
    di = cv2.dilate(th, kernel, iterations=1)
    save_image(str(out_dir), "09_morph_erode.jpg", er)
    save_image(str(out_dir), "10_morph_dilate.jpg", di)

    print(f"Done ✅ Outputs saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
