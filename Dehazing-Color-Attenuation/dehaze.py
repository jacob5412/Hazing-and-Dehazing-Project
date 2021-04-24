# Import all the necessary packages to your arsenal
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal as sig
import os

# import guidedfilter
# from guidedfilter import guidedfilter as gF


def guide(I, P, r, e):

    h, w = np.shape(I)
    window = np.ones((r, r)) / (r * r)

    meanI = sig.convolve2d(I, window, mode="same")
    meanP = sig.convolve2d(P, window, mode="same")

    corrI = sig.convolve2d(I * I, window, mode="same")
    corrIP = sig.convolve2d(I * P, window, mode="same")

    varI = corrI - meanI * meanI
    covIP = corrIP - meanI * meanP
    a = covIP / (varI + e)
    b = meanP - a * meanI

    meana = sig.convolve2d(a, window, mode="same")
    meanb = sig.convolve2d(b, window, mode="same")

    q = meana * I + meanb

    return q


def localmin(D, r=15):
    R = int(r / 2)
    imax = D.shape[0]
    jmax = D.shape[1]
    LM = np.zeros([imax, jmax])
    for i in np.arange(D.shape[0]):
        for j in np.arange(D.shape[1]):
            iL = np.max([i - R, 0])
            iR = np.min([i + R, imax])
            jT = np.max([j - R, 0])
            jB = np.min([j + R, jmax])
            # print(D[iL:iR+1,jT:jB+1].shape)
            LM[i, j] = np.min(D[iL : iR + 1, jT : jB + 1])
    return LM


def postprocessing(GD, I):
    # this will give indices of the columnised image GD
    flat_indices = np.argsort(GD, axis=None)
    R, C = GD.shape
    top_indices_flat = flat_indices[int(np.round(0.999 * R * C)) : :]
    top_indices = np.unravel_index(top_indices_flat, GD.shape)

    max_v_index = np.unravel_index(np.argmax(V[top_indices], axis=None), V.shape)
    I = I / 255.0
    A = I[max_v_index[0], max_v_index[1], :]
    print("Atmosphere A = (r, g, b)")
    print(A)

    beta = 1.0
    transmission = np.minimum(np.maximum(np.exp(-1 * beta * GD), 0.1), 0.9)
    # transmission = np.exp(-1*beta*GD)
    transmission3 = np.zeros(I.shape)
    transmission3[:, :, 0] = transmission
    transmission3[:, :, 1] = transmission
    transmission3[:, :, 2] = transmission

    J = A + (I - A) / transmission3
    J = J - np.min(J)
    J = J / np.max(J)
    return J


if __name__ == "__main__":
    import sys

    fn = sys.argv[1]

    filename = os.path.split(fn)[-1].split(".")[0]
    filepath = os.path.split(fn)[0]

    # Read the Image
    _I = cv2.imread(fn)
    # opencv reads any image in Blue-Green-Red(BGR) format,
    # so change it to RGB format, which is popular.
    I = cv2.cvtColor(_I, cv2.COLOR_BGR2RGB)
    # Split Image to Hue-Saturation-Value(HSV) format.
    H, S, V = cv2.split(cv2.cvtColor(_I, cv2.COLOR_BGR2HSV))
    V = V / 255.0
    S = S / 255.0

    # Calculating Depth Map using the linear model fit by ZHU et al.
    # Refer Eq(8) in mentioned research paper
    # Values given under EXPERIMENTS section
    theta_0 = 0.121779
    theta_1 = 0.959710
    theta_2 = -0.780245
    sigma = 0.041337
    epsilon = np.random.normal(0, sigma, H.shape)
    D = theta_0 + theta_1 * V + theta_2 * S + epsilon

    # saving depth map
    plt.imsave(os.path.join(filepath, filename + "_depth_map.jpg"), D)

    # Local Minima of Depth map
    LMD = localmin(D, 15)
    # LMD = D

    # Guided Filtering
    r = 8
    # try r=2, 4, 8 or 18
    eps = 0.2 * 0.2
    # try eps=0.1^2, 0.2^2, 0.4^2
    # eps *= 255 * 255;   # Because the intensity range of our images is [0, 255]
    GD = guide(D, LMD, r, eps)

    J = postprocessing(GD, I)

    # Plot the generated raw depth map
    # plt.subplot(121)
    plt.imshow(J)
    plt.title("Dehazed Image")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # save the depthmap.
    # Note: It will be saved as gray image.
    plt.imsave(os.path.join(filepath, filename + "_dehazed.jpg"), J)
