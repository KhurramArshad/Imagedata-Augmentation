from os import listdir
from os.path import isdir
from PIL import Image, ImageEnhance
import numpy as np

import cv2
import os

#Load images and extract faces for all images in a directory
def load_faces(directory):
    # enumerate files
    for filename in listdir(directory):
        #path
        fileName = directory + filename

        # get face
        try:
            saturation(fileName, directory)
            brightnessAndContrast(fileName, directory)
            transformation(fileName, directory)
        except Exception as e:
            continue

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    # enumerate folders , on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'

        #skip any file that might be in the dir
        if not isdir(path):
            continue

        # load all faces in the sub directory
        load_faces(path)

def brightnessAndContrast(fileName, directory):
    # 35 brightness image
    img = Image.open(fileName)
    enhancer = ImageEnhance.Brightness(img)
    brightnessImage = enhancer.enhance(2)
    brightnessImage.save(directory + "brightness1_Image.jpg")

    # 36 brightness image
    img = Image.open(fileName)
    enhancer = ImageEnhance.Brightness(img)
    brightnessImage = enhancer.enhance(1.5)
    brightnessImage.save(directory + "brightness2_Image.jpg")

    # 37 brightness image
    img = Image.open(fileName)
    enhancer = ImageEnhance.Brightness(img)
    brightnessImage = enhancer.enhance(1)
    brightnessImage.save(directory + "brightness3_Image.jpg")

    # 38 brightness image
    img = Image.open(fileName)
    enhancer = ImageEnhance.Brightness(img)
    brightnessImage = enhancer.enhance(0.5)
    brightnessImage.save(directory + "brightness4_Image.jpg")

    # 39 contrast image
    img = Image.open(fileName)
    enhancer = ImageEnhance.Contrast(img)
    brightnessImage = enhancer.enhance(2)
    brightnessImage.save(directory + "Contrast1_Image.jpg")

    # 40 contrast image
    img = Image.open(fileName)
    enhancer = ImageEnhance.Contrast(img)
    brightnessImage = enhancer.enhance(3)
    brightnessImage.save(directory + "Contrast2_Image.jpg")

    # 41 contrast image
    img = Image.open(fileName)
    enhancer = ImageEnhance.Contrast(img)
    brightnessImage = enhancer.enhance(4)
    brightnessImage.save(directory + "Contrast3_Image.jpg")

    # 42 contrast image
    img = Image.open(fileName)
    enhancer = ImageEnhance.Contrast(img)
    brightnessImage = enhancer.enhance(5)
    brightnessImage.save(directory + "Contrast4_Image.jpg")

def saturation(fileName, directory):
    # 43 saturation
    img = cv2.imread(fileName, 1)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # multiply by a factor to change the saturation
    hsvImg[..., 1] = hsvImg[..., 1] * 2
        # multiply by a factor less than 1 to reduce the brightnss
    hsvImg[..., 2] = hsvImg[...,2] * 0.6
    saturation = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    cv2.imwrite(directory + 'img_saturation1.jpg', saturation)

    # 44 saturation
    img = cv2.imread(fileName, 1)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # multiply by a factor to change the saturation
    hsvImg[..., 1] = hsvImg[..., 1] * 1.7
    # multiply by a factor less than 1 to reduce the brightnss
    hsvImg[..., 2] = hsvImg[..., 2] * 0.6
    saturation = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    cv2.imwrite(directory + 'img_saturation2.jpg', saturation)

    # 45 saturation
    img = cv2.imread(fileName, 1)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # multiply by a factor to change the saturation
    hsvImg[..., 1] = hsvImg[..., 1] * 2
    # multiply by a factor less than 1 to reduce the brightnss
    hsvImg[..., 2] = hsvImg[..., 2] * 1
    saturation = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    cv2.imwrite(directory + 'img_saturation3.jpg', saturation)

    # 46 saturation
    img = cv2.imread(fileName, 1)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # multiply by a factor to change the saturation
    hsvImg[..., 1] = hsvImg[..., 1] * -1
    # multiply by a factor less than 1 to reduce the brightnss
    hsvImg[..., 2] = hsvImg[..., 2] * 0.6
    saturation = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    cv2.imwrite(directory + 'img_saturation4.jpg', saturation)

    # 47 saturation
    img = cv2.imread(fileName, 1)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # multiply by a factor to change the saturation
    hsvImg[..., 1] = hsvImg[..., 1] * -0.5
    # multiply by a factor less than 1 to reduce the brightnss
    hsvImg[..., 2] = hsvImg[..., 2] * 0.6
    saturation = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    cv2.imwrite(directory + 'img_saturation5.jpg', saturation)

    # 48 saturation
    img = cv2.imread(fileName, 1)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # multiply by a factor to change the saturation
    hsvImg[..., 1] = hsvImg[..., 1] * -0.5
    # multiply by a factor less than 1 to reduce the brightnss
    hsvImg[..., 2] = hsvImg[..., 2] * 1
    saturation = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    cv2.imwrite(directory + 'img_saturation6.jpg', saturation)

def transformation(fileName, directory):
    # 42 gamma
    img = cv2.imread(fileName, 1)
    img = img/255.0
    img = cv2.pow(img, 3)
    gammaImage = np.uint8(img * 255)
    cv2.imwrite(directory + 'image_gamma.jpg', gammaImage)

    # 1. rotate 10 to left
    img = cv2.imread(fileName, 1)
    rows,cols = img.shape[:2]
    Matrix = cv2.getRotationMatrix2D((cols/2,rows/2), 10, 1)
    rotated10 = cv2.warpAffine(img, Matrix, (cols, rows))
    cv2.imwrite(directory + 'img_rotate_10_left.jpg', rotated10)

    # 2. rotate 10 to right
    img = cv2.imread(fileName, 1)
    rows, cols = img.shape[:2]
    Matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -10, 1)
    rotated10 = cv2.warpAffine(img, Matrix, (cols, rows))
    cv2.imwrite(directory + 'img_rotate_10_right.jpg', rotated10)

    # 3. rotate 15 to left
    img = cv2.imread(fileName, 1)
    rows, cols = img.shape[:2]
    Matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    rotated15 = cv2.warpAffine(img, Matrix, (cols, rows))
    cv2.imwrite(directory + 'img_rotate_15_left.jpg', rotated15)

    # 4. rotate 15 to right
    img = cv2.imread(fileName, 1)
    rows, cols = img.shape[:2]
    Matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
    rotated15 = cv2.warpAffine(img, Matrix, (cols, rows))
    cv2.imwrite(directory + 'img_rotate_15_right.jpg', rotated15)

    # 5. rotate 20 to left
    img = cv2.imread(fileName, 1)
    rows, cols = img.shape[:2]
    Matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 20, 1)
    rotated20 = cv2.warpAffine(img, Matrix, (cols, rows))
    cv2.imwrite(directory + 'img_rotate_20_left.jpg', rotated20)

    # 6. rotate 20 to right
    img = cv2.imread(fileName, 1)
    rows, cols = img.shape[:2]
    Matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -20, 1)
    rotated20 = cv2.warpAffine(img, Matrix, (cols, rows))
    cv2.imwrite(directory + 'img_rotate_20_right.jpg', rotated20)

    # 7. Flip
    img = cv2.imread(fileName, 1)
    img_flip = cv2.flip(img, 1)
    cv2.imwrite(directory + 'img_flip.jpg', img_flip)

    # 8. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 6, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_1.jpg', BilateralBlur)

    # 9. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 7, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_2.jpg', BilateralBlur)

    # 10. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 8, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_3.jpg', BilateralBlur)

    # 11. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_4.jpg', BilateralBlur)

    # 12. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 10, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_5.jpg', BilateralBlur)

    # 13. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 11, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_6.jpg', BilateralBlur)

    # 14. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 12, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_7.jpg', BilateralBlur)

    # 15. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 13, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_8.jpg', BilateralBlur)

    # 16. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 14, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_9.jpg', BilateralBlur)

    # 17. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 15, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_10.jpg', BilateralBlur)

    # 18. blur bilateral
    img = cv2.imread(fileName, 1)
    BilateralBlur = cv2.bilateralFilter(img, 30, 75, 75)
    cv2.imwrite(directory + 'img_blur_bilateral_11.jpg', BilateralBlur)

    # 19. gaussian Blur
    img = cv2.imread(fileName, 1)
    gaussianBlur = cv2.GaussianBlur(img, (5,5), 10000)
    cv2.imwrite(directory + 'img_blur_gaussian.jpg', gaussianBlur)

    # 20. median Blur
    img = cv2.imread(fileName, 1)
    medianBlur = cv2.medianBlur(img, 5)
    cv2.imwrite(directory + 'img_blur_median.jpg', medianBlur)

    # 21. Blur
    img = cv2.imread(fileName, 1)
    blur = cv2.blur(img, (5, 5))
    cv2.imwrite(directory + 'img_blur.jpg', blur)

    # 22. blue image
    img = cv2.imread(fileName, 1)
    blueImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(directory + 'img_blue_color.jpg', blueImage)

    # 23. gray image
    img = cv2.imread(fileName, 1)
    grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(directory + 'img_gray_color.jpg', grayImage)

    # 24. black and white image
    img = cv2.imread(fileName, 1)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(directory + 'img_blackAndWhiteImage.jpg', blackAndWhiteImage)

    # 25. yellow image
    img = cv2.imread(fileName, 1)
    yellowImage = cv2.cvtColor(img, cv2.COLOR_XYZ2RGB)
    cv2.imwrite(directory + 'img_yellow_color.jpg', yellowImage)

    # 29. color image
    img = cv2.imread(fileName, 1)
    colorImage = cv2.cvtColor(img, cv2.COLOR_XYZ2RGB)
    cv2.imwrite(directory + 'img_color_4.jpg', colorImage)

    # 30. color image
    img = cv2.imread(fileName, 1)
    colorImage = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
    cv2.imwrite(directory + 'img_color_5.jpg', colorImage)

    # 32. color image
    img = cv2.imread(fileName, 1)
    colorImage = cv2.cvtColor(img, cv2.COLOR_LRGB2Luv)
    cv2.imwrite(directory + 'img_color_7.jpg', colorImage)

    # 26. color image
    img = cv2.imread(fileName, 1)
    colorImage = cv2.cvtColor(img, cv2.COLOR_YUV420sp2BGRA)
    cv2.imwrite(directory + 'img_color_1.jpg', colorImage)

load_dataset('../images/train/')



