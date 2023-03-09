import cv2
import imutils
import pytesseract
from skimage.util.arraycrop import crop
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# Reading the image file


def predict(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    cv2.imshow("Orignal Image", image)
    # Image Conversion to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Scale Image", gray)
    cv2.waitKey(0)

    # now we will reduce noise from the image and make it smooth

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("Smoother Image", gray)
    cv2.waitKey(0)

    # so now we will find the edges of the images
    edged = cv2.Canny(gray, 170, 200)
    cv2.imshow("Canny edge", edged)
    cv2.waitKey(0)

    # Find the contours based on the images
    cntns, new = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Copy orignal image to draw all the contours
    image1 = image.copy()
    # this values are fixed to draw all the contours in an image
    cv2.drawContours(image1, cntns, -1, (0, 255, 0), 3)
    cv2.imshow("Canny after contouring", image1)
    cv2.waitKey(0)

    # Reverse the order of sorting

    cntns = sorted(cntns, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCount = 0

    # Because currently we don't have any contour or you can say it will show how many number plates are there in image
    # To draw top 30 contours  we will  make copy of original image and use
    # Use because we don't want to edit anything in our original image

    image2 = image.copy()
    cv2.drawContours(image2, cntns, -1, (0, 255, 0), 3)
    cv2.imshow("Top 30 contours", image2)
    cv2.waitKey(0)

    # We will run a loop on our contours to find the best possible contour of our expectes number plate
    count = 0
    name = 1  # name of our cropped image

    for i in cntns:
        perimeter = cv2.arcLength(i, True)
        # perimeter is also called as arclengthand we can find directly in python  using arclenght function
        approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
        # approxPolyDP we have used because it approximates the curve of polygon with the precision
        if(len(approx) == 4):    # 4 means it has 4 corner which will be most probably our number plate as it also has 4 corner
            NumberPlateCount = approx
            # now we will crop that rectangle part
            x, y, w, h = cv2.boundingRect(i)
            crp_img = image[y:y+h, x:x+w]

            cv2.imwrite("cropped" + '.jpg', crp_img)
            name += 1

            break

    # now we will draw contour in our main image that we have identified as a number plate
    cv2.drawContours(image, [NumberPlateCount], -1, (0, 255, 0), 3)
    cv2.imshow("Final image", image)
    cv2.waitKey(0)

    # We will crop only the part of number plate
    crop_img_loc = 'cropped.jpg'
    cv2.imshow("cropped image", cv2.imread(crop_img_loc))
    cv2.waitKey(0)
    crop_img_loc = cv2.imread("cropped.jpg")
    hsv = cv2.cvtColor(crop_img_loc, cv2.COLOR_BGR2HSV)

    msk = cv2.inRange(hsv, np.array([0,0,0]), np.array([179,255,154]))

    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, msk)
    cv2.imshow("new", res)
    cv2.waitKey(0)

# OCR
    txt = pytesseract.image_to_string(crop_img_loc, config="--psm 6")



    return(txt)