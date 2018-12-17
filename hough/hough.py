import math
import sys

import cv2 as cv
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def main(argv):
    default_file = "input.tiff"
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    src = cv.resize(src, (1024, 1024))
    dst = cv.Canny(src, 300, 200, None, 3)
    # dst = cv.Canny(src,50,150,apertureSize = 3)
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.zeros_like(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 350, None, 0, 0)
    # lines = cv.HoughLines(dst,1,np.pi/180,200)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            print(rho)
            print((theta))
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 2000 * (a)))
            pt2 = (int(x0 - 1300 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdstP, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)
            # plt.imshow(cdstP)
            # plt.show()
    gray = cv.cvtColor(cdstP, cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    #
    # linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    #
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    # cv.imshow("Source.png", src)
    # cv.imwrite("Source.png", src)
    # cdstP[dst>0]=[0,255,0]
    # cv.imwrite("output.png", cdstP)
    # plt.imshow(cdstP)
    # plt.show()

    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv.imwrite("Standar Hough Line Transform", cdst, [cv.IMWRITE_JPEG_QUALITY,90])
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)


if __name__ == "__main__":
    main(sys.argv[1:])
