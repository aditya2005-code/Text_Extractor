import os
import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract as pt
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Set Tesseract path (Windows)
pt.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Flask setup
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ===== Helper functions =====
def grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def noise_removal(img):
    kernel = np.ones((1, 1), np.uint8)
    img = cv.dilate(img, kernel, iterations=1)
    img = cv.erode(img, kernel, iterations=1)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    return cv.medianBlur(img, 3)

def needs_inversion(path, threshold=127):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    dark = np.sum(img < threshold)
    bright = np.sum(img >= threshold)
    return bright > dark

def needs_deskew(img, skew_threshold=1.0):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    return abs(angle) > skew_threshold

def getSkewAngle(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=2)
    contours, _ = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    angle = cv.minAreaRect(contours[0])[-1]
    if angle < -45: angle += 90
    return -angle

def rotateImage(img, angle):
    (h, w) = img.shape[:2]
    M = cv.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

def deskew(img):
    angle = getSkewAngle(img)
    if abs(angle) < 2.0: return img
    return rotateImage(img, -angle)

def remove_borders(img):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    cnt = sorted(contours, key=cv.contourArea)[-1]
    x, y, w, h = cv.boundingRect(cnt)
    return img[y:y+h, x:x+w]

def needs_thinning_or_thickening(path, min_width=2, max_width=6):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    _, bw = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    dist = cv.distanceTransform(bw, cv.DIST_L2, 5)
    mean_stroke = np.mean(dist)
    if mean_stroke < min_width: return "thin"
    elif mean_stroke > max_width: return "thick"
    else: return "ok"

# ===== Main processing =====
def process_image(file_path):
    img = cv.imread(file_path)
    if img is None: raise FileNotFoundError(f"Image not found: {file_path}")

    if needs_deskew(img): img = deskew(img)
    if needs_inversion(file_path): img = cv.bitwise_not(img)
    img = grayscale(img)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 15, 8)
    img = noise_removal(img)
    result = needs_thinning_or_thickening(file_path)
    if result == "thin":
        img = cv.bitwise_not(img)
        img = cv.erode(img, np.ones((2,2),np.uint8), iterations=1)
        img = cv.bitwise_not(img)
    elif result == "thick":
        img = cv.bitwise_not(img)
        img = cv.dilate(img, np.ones((2,2),np.uint8), iterations=1)
        img = cv.bitwise_not(img)
    img = remove_borders(img)
    img = cv.copyMakeBorder(img, 150, 150, 150, 150, cv.BORDER_CONSTANT, value=[255,255,255])
    cv.imwrite("assets/debug_image.jpg", img)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    custom_config = r'--oem 3 --psm 6'
    text = pt.image_to_string(img_rgb, config=custom_config)
    print("Extracted Text:", text)
    return text.strip()

# ===== Flask routes =====
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/extract", methods=["POST"])
def extract_text():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        text = process_image(file_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
