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
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

def needs_inversion(image_path, threshold=127):
    """
    Check if an image needs inversion for OCR text extraction.
    Returns True if inversion is recommended, False otherwise.
    """
    # Read image in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Count dark vs bright pixels
    dark_pixels = np.sum(img < threshold)
    bright_pixels = np.sum(img >= threshold)

    # If background is mostly dark, inversion NOT needed
    # If background is mostly bright, inversion NEEDED
    return dark_pixels > bright_pixels  # True = needs inversion

#Gray Scaling
def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#noise removal
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)
    return (image)

#font thickening or thinning check
def needs_thinning_or_thickening(image_path, min_width=2, max_width=6):
    """
    Check if text in an image needs thinning (erosion) or thickening (dilation).
    Returns: "thin", "thick", or "ok"
    """
    # Read as grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Threshold to binary
    _, bw = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)

    # Distance transform (estimates stroke width)
    dist = cv.distanceTransform(bw, cv.DIST_L2, 5)
    mean_stroke = np.mean(dist)

    if mean_stroke < min_width:
        return "thin"   # needs thickening
    elif mean_stroke > max_width:
        return "thick"  # needs thinning
    else:
        return "ok"


#deskewing angle detection

def needs_deskew(cvImage, skew_threshold=1.0):
    gray = cv.cvtColor(cvImage, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv.minAreaRect(coords)[-1]

    # Fix OpenCV angle range
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # SAFETY FIX: If angle ≈ ±90, ignore
    if abs(abs(angle) - 90) < 2:
        angle = 0.0

    print(f"[Detected angle]: {angle}")
    return abs(angle) > skew_threshold



def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv.cvtColor(newImage, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    for c in contours:
        rect = cv.boundingRect(c)
        x,y,w,h = rect
        cv.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv.minAreaRect(largestContour)
    cv.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv.warpAffine(newImage, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage, angle_threshold: float = 2.0):
    angle = getSkewAngle(cvImage)
    print(f"Deskew angle: {angle:.2f}°")
    if abs(angle) < angle_threshold:
        print("Skipping rotation (image is already straight).")
        return cvImage  # don’t rotate
    return rotateImage(cvImage, -1.0 * angle)

def remove_borders(image):
    contours, heiarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

def process_image(image_path):
    image_file = image_path
    img = cv.imread(image_file)
    
    if needs_deskew(img):
        img = deskew(img)
    needInverted = needs_inversion(image_file)
    if(needInverted):
        img = cv.bitwise_not(img)
    
    img = grayscale(img)
    thresh, im_bw = cv.threshold(img, 210, 230, cv.THRESH_BINARY)
    img = im_bw
    img = noise_removal(img)
    
    result = needs_thinning_or_thickening("assets/no_noise.jpg")
    if(result == "thin"):
        img = cv.bitwise_not(img)
        kernel = np.ones((2,2),np.uint8)
        img = cv.erode(img, kernel, iterations=1)
        img = cv.bitwise_not(img)
    elif (result == "thick"):
        img = cv.bitwise_not(img)
        kernel = np.ones((2,2),np.uint8)
        img = cv.dilate(img, kernel, iterations=1)
        img = cv.bitwise_not(img)
    else:
        pass
    
    img = remove_borders(img)
    
    color = [255, 255, 255]
    top, bottom, left, right = [150]*4
    
    image_with_border = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    img = image_with_border
    
    ocr_result = pt.image_to_string(img)
    return ocr_result


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
