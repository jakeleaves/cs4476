import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io


class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        ###### START CODE HERE ######
        self.img = plt.imread('inputPS1Q3.jpg')
        self.imgA = np.asarray(self.img)
        ###### END CODE HERE ######
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image
        """
        ###### START CODE HERE ######
        r = self.img[:, :, 0]
        g = self.img[:, :, 1]
        b = self.img[:, :, 2]
        gray = r*0.299 + g*0.587 + b*0.114
        ###### END CODE HERE ######
        return gray
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """

        ###### START CODE HERE ######
        r = self.imgA[:, :, 0]
        g = self.imgA[:, :, 1]
        swapImg = self.img.copy()
        swapImg[:, :, 0] = g
        swapImg[:, :, 1] = r

        plt.figure()
        plt.imshow(swapImg)
        plt.show()

        ###### END CODE HERE ######
        return swapImg

    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        ###### START CODE HERE ######
        grayImg = self.rgb2gray(self.img.copy())
        grayImg = grayImg.astype(np.uint8)

        plt.figure()
        plt.imshow(grayImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return grayImg
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        negativeImg = 255 - grayImg

        negativeImg = negativeImg.astype(np.uint8)

        plt.figure()
        plt.imshow(negativeImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        mirrorImg = grayImg[:, :: -1]
        mirrorImg = mirrorImg.astype(np.uint8)

        plt.figure()
        plt.imshow(mirrorImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """

        ###### START CODE HERE ######
        grayImg = self.prob_3_2().astype(np.double)
        mirrorImg = self.prob_3_4().astype(np.double)
        avgImg = ((mirrorImg + grayImg)/2).astype(np.uint8)

        plt.figure()
        plt.imshow(avgImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            noisyImg, noise: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
            and the noise
        """
        ###### START CODE HERE ######
        Z = np.zeros((900, 600), dtype='double')
        R = np.random.randint(0, 255, (900, 600), dtype='uint8').astype(np.double)
        noise = Z + R
        noisyImg = np.clip(self.prob_3_2() + noise, 0, 255).astype(np.uint8)

        plt.figure()
        plt.imshow(noisyImg, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return noisyImg, noise
        
        
if __name__ == '__main__': 
    
    p3 = Prob3()
    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    noisyImg,_ = p3.prob_3_6()

    




