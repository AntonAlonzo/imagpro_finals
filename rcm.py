import os
import numpy as np
import cv2 as cv

# "Constant" declaration of Robinsons Compass Mask kernels
RCM = [
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # East
    np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),  # South East
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),  # South
    np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),  # South West
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),  # West
    np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),  # North West
    np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  # North
    np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])   # North East
]

# Function to create a folder to hold the different compass outputs for a single image input
def createDirectory(name):

    path = "./compasses/" + name

    if not os.path.exists(path):
        os.mkdir(path)
        print("Folder for %s data created" % path)
    else:
        print("Folder for %s data already exists" % path)

def applyMasks(img, name, RCM=RCM):
    gradient_images = [cv.filter2D(img, -1, mask) for mask in RCM]
    absolute_gradients = [np.abs(gradient) for gradient in gradient_images]

    # TODO: use createDirectory to make folder to store all compass data of the img

    return absolute_gradients
    

def detectEdge(img):

    abs_gradients = applyMasks(img)
    img_edged = np.sum(abs_gradients, axis = 0)

    # TODO: write image to final folder
    
    return img_edged 