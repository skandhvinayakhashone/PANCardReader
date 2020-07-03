import os
from imutils import paths
import argparse
import cv2
import pytesseract
import os
from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import sys
import subprocess


def blur(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm > 1000:
        return bool(0)
    else:
        return bool(1)

def pixelated(imagePath):
    im = Image.open(imagePath)
    im2 = im.transform(im.size, Image.AFFINE, (1,0,1,0,1,1))
    im3 = ImageChops.subtract(im, im2)
    im3 = np.asarray(im3)
    im3 = np.sum(im3,axis=0)[:-1]
    mean = np.mean(im3)
    peak_spacing = np.diff([i for i,v in enumerate(im3) if (all(x > mean*2 for x in v))])
    mean_spacing = np.mean(peak_spacing)
    std_spacing = np.std(peak_spacing)
    if mean_spacing < std_spacing:
        return bool(0)
    else:
        return bool(1)

if __name__=="__main__":
    Path = os.listdir("./pancard")
    for imagePath in Path:
        Blur=blur(imagePath)
        Pixelated=pixelated(imagePath)
        if Blur:
            print("The Image is Blur")
        if Pixelated:
            print("The Image is Pixelated")
        elif not(Blur or Pixelated):
            cmd= 'tesseract '+ str(imagePath) +' stdout --dpi 300 --psm 1 --oem 1'
            output = subprocess.getoutput(cmd)
            print (output)

            #processign
            img=imagePath
            im = Image.open(img)

            rgb_im = im.convert('RGB')
            rgb_im.save("test.jpg", dpi=(300,300))

            image_dpi = cv2.imread('test.jpg',0)
            os.remove("test.jpg")
            blur = cv2.bilateralFilter(image_dpi,15,75,75)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            ret3,th3 = cv2.threshold(th3,0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)
            imagem = cv2.bitwise_not(th3)
            kernel = np.ones((1,1),np.uint8)
            eroded_img = cv2.erode(imagem,kernel,iterations = 1)
            canny=cv2.Canny(imagem,100,200)
            #cv2.imwrite('/bw_image.jpg', eroded_img)

            pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.0.0/bin/tesseract'
            config = ('-l eng --oem 1 --psm 3')
            text = pytesseract.image_to_string(eroded_img, config=config)
            print(text)
