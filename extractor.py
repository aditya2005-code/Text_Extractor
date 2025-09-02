import pytesseract as pt
from PIL import Image
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable cross-origin requests

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Display method for debugging
def display(im_path):
    import matplotlib.pyplot as plt
    dpi = 80
    im_data = plt.imread(im_path)
    height, width  = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()

# Inversion check
def needs_inversion(image_path, threshold=127):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    dark_pixels = np.sum(img < threshold)
    bright_pixels = np.sum(img >= threshold)
    return bright_pixels > dark_pixels  # True if mostly bright → needs inversion

# Grayscale
def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Noise removal
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)
    return image

# Font thinning/thickening check
def needs_thinning_or_thickening(image_path, min_width=2, max_width=6):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, bw = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    dist = cv.distanceTransform(bw, cv.DIST_L2, 5)
    mean_stroke = np.mean(dist)
    if mean_stroke < min_width:
        return "thin"
    elif mean_stroke > max_width:
        return "thick"
    else:
        return "ok"

# Deskewing
def needs_deskew(cvImage, skew_threshold=1.0):
    gray = cv.cvtColor(cvImage, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(abs(angle) - 90) < 2:
        angle = 0.0
    return abs(angle) > skew_threshold

def getSkewAngle(cvImage) -> float:
    newImage = cvImage.copy()
    gray = cv.cvtColor(newImage, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=2)
    contours, _ = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    if not contours:
        return 0.0
    minAreaRect = cv.minAreaRect(contours[0])
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(newImage, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

def deskew(cvImage, angle_threshold: float = 2.0):
    angle = getSkewAngle(cvImage)
    if abs(angle) < angle_threshold:
        return cvImage
    return rotateImage(cvImage, -1.0 * angle)

def remove_borders(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = sorted(contours, key=cv.contourArea)[-1]
    x, y, w, h = cv.boundingRect(cnt)
    return image[y:y+h, x:x+w]

# Image preprocessing
def process_image(file_path):
    img = cv.imread(file_path)
    if needs_deskew(img):
        img = deskew(img)
        cv.imwrite("assets/deskewed.jpg", img)
        # display("assets/deskewed.jpg")  # optional for debug

    if needs_inversion(file_path):
        img = cv.bitwise_not(img)
        cv.imwrite("assets/inverted.jpg", img)
        # display("assets/inverted.jpg")  # optional

    img = grayscale(img)
    cv.imwrite("assets/gray.jpg", img)

    _, im_bw = cv.threshold(img, 210, 255, cv.THRESH_BINARY)
    cv.imwrite("assets/bw.jpg", im_bw)
    img = im_bw

    img = noise_removal(img)
    cv.imwrite("assets/no_noise.jpg", img)

    result = needs_thinning_or_thickening("assets/no_noise.jpg")
    if result == "thin":
        img = cv.bitwise_not(img)
        img = cv.erode(img, np.ones((2,2),np.uint8), iterations=1)
        img = cv.bitwise_not(img)
    elif result == "thick":
        img = cv.bitwise_not(img)
        img = cv.dilate(img, np.ones((2,2),np.uint8), iterations=1)
        img = cv.bitwise_not(img)

    img = remove_borders(img)
    cv.imwrite("assets/no_borders.jpg", img)

    # Add border
    img = cv.copyMakeBorder(img, 150, 150, 150, 150, cv.BORDER_CONSTANT, value=[255,255,255])
    cv.imwrite("assets/image_with_border.jpg", img)

    # OCR
    img = Image.open("assets/image_with_border.jpg")
    text = pt.image_to_string(img)
    return text.strip()

# Flask routes
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

    text = process_image(file_path)
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
