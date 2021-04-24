import cv2
import math
import numpy as np
import os


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)  # split image to r, g, b channels
    # finding color channel (r, g or b) with lowest intensity (low intensity contributed due to airlight)
    # dark pixels can directly provide an accurate estimation of the haze transmission
    dc = cv2.min(cv2.min(r, g), b)
    # return a rectangular structuring element of the specified size and shape for morphological operations
    # sz is the size of the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    # erodes image of rectangular sturcturing element of size 'sz'
    # this is the dark channel
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    # take height and width of image (first two columns)
    [h, w] = im.shape[:2]
    # image size would be height * width
    imsz = h * w

    # If we select the brightest 0.1% of the dark channel we will get the haziest pixels
    # Note: x*(0.1/100) = x/1000
    # any image can have a minimum of only 1 pixel or 0.1% brightest of dark channel as 1 pixel
    numpx = int(max(math.floor(imsz / 1000), 1))

    # reshape as 1 column array of size 'imsz'
    darkvec = dark.reshape(imsz, 1)
    # reshape as 3 column array of size 'imsz'
    imvec = im.reshape(imsz, 3)

    # return indices that would sort the array (descending order)
    # Since the lightest regions of the dark channel correspond to the haziest part of the original image.
    indices = darkvec.argsort()
    # taking those brightest pixels from the array
    indices = indices[imsz - numpx : :]

    atmsum = np.zeros([1, 3])  # return array of zeros
    for ind in range(1, numpx):
        # Switching back to the original RGB values of these same pixels
        # we can take the brightest as the atmospheric light
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    # Return a new array of given shape and type, without initializing entries.
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


if __name__ == "__main__":
    import sys

    fn = sys.argv[1]

    filename = os.path.split(fn)[-1].split(".")[0]
    filepath = os.path.split(fn)[0]

    src = cv2.imread(fn)
    # read the image

    # use astype to cast image to float 64 values
    I = src.astype("float64") / 255
    # normalizing the data to 0 - 1 (Since 255 is the maximum value)

    dark = DarkChannel(I, 15)
    # extracting dark channel prior
    A = AtmLight(I, dark)
    # extracting global atmospheric lighting
    # Transmission is an estimate of how much of the light from the
    # original object is making it through the haze at each pixel
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    # atmospheric light is subtracted from each pixel in proportion to the transmission at that pixel.
    J = Recover(I, t, A, 0.1)

    cv2.imshow("dark", dark)
    # dark channel
    cv2.imshow("t", t)
    # Transmission
    cv2.imshow("I", src)
    # original image
    cv2.imshow("J", J)
    # image after dehazing
    cv2.imwrite(os.path.join(filepath, filename + "_dehazed.png"), J * 255)
    cv2.waitKey()
