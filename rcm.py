import os
import numpy as np
import cv2 as cv

# "Constant" declaration of Robinsons Compass Mask kernels
RCM_KERNELS = [
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # East
    np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),  # South East
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),  # South
    np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),  # South West
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),  # West
    np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),  # North West
    np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  # North
    np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])   # North East
]

# "Constant" declaration of file name suffixes for RCM output
RCM_EXTS = [
    "E", "SE", "S", "SW", "W", "NW", "N", "NE"
]

#######################################################################################################
#
#   Function to create a folder to hold the different outputs of an input image
#
#   Parameters:
#       name    file name of data (to become folder name)
#
#   Returns:
#       file paths
#
#######################################################################################################
def createDirectory(name):

    rcm_path = "compasses/" + name
    out_path = "final/" + name

    if not os.path.exists(rcm_path):
        os.mkdir(rcm_path)
        print("RCM folder for %s data created" % rcm_path)
    else:
        print("RCM folder for %s data already exists" % rcm_path)

    return rcm_path, out_path

#######################################################################################################
#
#   Function to write eight (8) different image files representing the different RCM kernel output
#   for a specfic image
#
#   Parameters:
#       name    file name of image (without extension)
#       path    directory to upload output
#       abs     absolute gradients of RCM kernel output
#
#######################################################################################################
def generateRcmOutputs(name, path, abs, exts=RCM_EXTS):

    # !! CONFIIGURE FILE TYPE HERE !!
    file_type = ".png"

    # Saving numpy file
    np.save(os.path.join(path, name + "_rcm_outputs.npy"), abs)
    
    for i in range(len(abs)):
       img_path = os.path.join(path, name + "_" + exts[i] + file_type)
       cv.imwrite(img_path)
    
    

#######################################################################################################
#
#   Function to apply the RCM kernels to the image
#
#   Parameters:
#       img     image data
#       name    file name of data (without extension)
#       RCM     constantly set RCM
#
#   Returns:
#       Absolute gradients of the kernel outputs
#
#######################################################################################################
def applyMasks(img, name, RCM=RCM_KERNELS):
    print(RCM)
    gradient_images = [cv.filter2D(img, -1, mask) for mask in RCM]
    absolute_gradients = [np.abs(gradient) for gradient in gradient_images]

    # Create unique directory for image to hold RCM indiv data
    rcm_path, _ = createDirectory(name)

    # Generate image files of RCM kernel output
    generateRcmOutputs(name, rcm_path, absolute_gradients)

    return absolute_gradients
    
#######################################################################################################
#
#   Function to generate the final edge-detected image data
#
#   Parameters:
#       img     image data
#
#   Returns:
#       Post-processed image data
#
#######################################################################################################
def detectEdge(img, name):

    abs_gradients = applyMasks(img)
    img_edged = np.sum(abs_gradients, axis = 0)

    # TODO: write image to final folder
    output_f = "output/" + name
    os.makedirs(output_f, exist_ok=True)

    # Save np
    np.save(os.path.join(output_f, name + "_final_output.npy"), img_edged)

    # Save image
    cv.imwrite(os.path.join(output_f, name + "_final_output.png"), img_edged)
    
    return img_edged 