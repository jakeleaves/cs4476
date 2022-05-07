import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, io


class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""

        ###### START CODE HERE ######
        self.indoor = plt.imread('indoor.png')
        self.outdoor = plt.imread('outdoor.png')
        ###### END CODE HERE ######

    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""

        ###### START CODE HERE ######
        ri = self.indoor[:, :, 0]
        gi = self.indoor[:, :, 1]
        bi = self.indoor[:, :, 2]

        ro = self.outdoor[:, :, 0]
        go = self.outdoor[:, :, 1]
        bo = self.outdoor[:, :, 2]

        plt.figure()
        plt.title("Outdoor Red")
        plt.imshow(ro, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Outdoor Green")
        plt.imshow(go, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Outdoor Blue")
        plt.imshow(bo, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Indoor Red")
        plt.imshow(ri, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Indoor Green")
        plt.imshow(gi, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Indoor Blue")
        plt.imshow(bi, cmap='gray', aspect='auto')
        plt.show()

        # LAB color space
        indoor_lab = cv2.cvtColor(self.indoor, cv2.COLOR_RGB2LAB)
        outdoor_lab = cv2.cvtColor(self.outdoor, cv2.COLOR_RGB2LAB)

        ril = indoor_lab[:, :, 0]
        gil = indoor_lab[:, :, 1]
        bil = indoor_lab[:, :, 2]

        rol = outdoor_lab[:, :, 0]
        gol = outdoor_lab[:, :, 1]
        bol = outdoor_lab[:, :, 2]

        plt.figure()
        plt.title("Outdoor Lab Red")
        plt.imshow(rol, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Outdoor Lab Green")
        plt.imshow(gol, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Outdoor Lab Blue")
        plt.imshow(bol, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Indoor Lab Red")
        plt.imshow(ril, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Indoor Lab Green")
        plt.imshow(gil, cmap='gray', aspect='auto')
        plt.show()

        plt.figure()
        plt.title("Indoor Lab Blue")
        plt.imshow(bil, cmap='gray', aspect='auto')
        plt.show()
        ###### END CODE HERE ######
        return

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """

        ###### START CODE HERE ######
        img = io.imread('inputPS1Q4.jpg') / 255.0

        # Value
        V = img.max(2)

        # Saturation
        m = img.min(2)
        C = V - m
        S = np.copy(C)
        S[V != 0.0] = C[V != 0.0] / V[V != 0.0]
        S[V == 0.0] = 0.0

        # Hue
        H = np.copy(S)

        row = img.shape[0]
        col = img.shape[1]
        for x in range(row):
            for y in range(col):
                # H' calculation
                if C[x][y] == 0:
                    H[x][y] = 0
                elif V[x][y] == img[x][y][0]:
                    H[x][y] = (img[x][y][1] - img[x][y][2]) / C[x][y]
                elif V[x][y] == img[x][y][1]:
                    H[x][y] = (img[x][y][2] - img[x][y][0]) / C[x][y] + 2.0
                elif V[x][y] == img[x][y][2]:
                    H[x][y] = (img[x][y][0] - img[x][y][1]) / C[x][y] + 4.0
                # H calculation
                if H[x][y] < 0:
                    H[x][y] = H[x][y] / 6 + 1
                else:
                    H[x][y] = H[x][y] / 6

        HSV = np.dstack((np.dstack((H, S)), V))

        plt.imsave('outputPS1Q4.jpg', HSV)
        ###### END CODE HERE ######
        return HSV


if __name__ == '__main__':
    p4 = Prob4()
    p4.prob_4_1()
    HSV = p4.prob_4_2()
