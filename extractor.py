import pytesseract as pst
import cv2 as cv
import numpy as np
from PIL import Image

def is_light_text_on_dark(img):
    mean_intensity = np.mean(img)
    return mean_intensity < 127 

def greyscale_img(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def inverted_img(img):
    inverted = cv.bitwise_not(img)
    cv.imwrite("assets/inverted.png", inverted)
    return inverted

def getSkewAngle(cvImage) -> float:
    gray = cv.cvtColor(cvImage, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=2)

    contours, _ = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    minAreaRect = cv.minAreaRect(max(contours, key=cv.contourArea))
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(cvImage, angle: float):
    (h, w) = cvImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(cvImage, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)
    return image

def estimate_stroke_width(binary_img):
    dist = cv.distanceTransform(binary_img, cv.DIST_L2, 3)
    if np.count_nonzero(dist) == 0:
        return 0
    return np.mean(dist[dist > 0]) * 2

def thin_font(image):
    image = cv.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    return cv.bitwise_not(image)

def thick_font(image):
    image = cv.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    return cv.bitwise_not(image)

def remove_borders(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv.boundingRect(max(contours, key=cv.contourArea))
    return image[y:y+h, x:x+w]

# ---------------- Main pipeline ----------------

img_file = "assets/page_01.jpg"
img = cv.imread(img_file)

greyscaleimg = greyscale_img(img)
cv.imwrite("assets/grey_scale.png", greyscaleimg)

if is_light_text_on_dark(greyscaleimg):
    greyscaleimg = inverted_img(greyscaleimg)

# deskew
fixed = deskew(img)
cv.imwrite("assets/fixed.png", fixed)

img = greyscale_img(fixed)
cv.imwrite("assets/greyscale_fixed.png", img)

# Binary threshold
_, im_bw = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite("assets/bw_image.jpg", im_bw)

no_noise = noise_removal(im_bw)
cv.imwrite("assets/no_noise.jpg", no_noise)

# stroke width check
stroke = estimate_stroke_width(no_noise)
if stroke < 2:
    processed = thick_font(no_noise)
else:
    processed = thin_font(no_noise)
cv.imwrite("assets/processed.png", processed)

no_borders = remove_borders(processed)
cv.imwrite("assets/no_borders.png", no_borders)

# add padding
color = [255, 255, 255]
top, bottom, left, right = [150]*4
image_with_border = cv.copyMakeBorder(no_borders, top, bottom, left, right,
                                      cv.BORDER_CONSTANT, value=color)
cv.imwrite("assets/image_with_border.jpg", image_with_border)

# OCR
text = pst.image_to_string(Image.open("assets/image_with_border.jpg"),
                            lang='eng', config='--psm 6')
print(text)
