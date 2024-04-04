# Plate Number Recognition System using Robinsons Compass Mask Edge Detection | RCM Header File
# IMAGPRO | Alonzo, Hernandez, Solis, Susada

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
#######################################################################################################
def createDirectories(name):

    rcm_path = "compasses\\" + name
    out_path = "output\\" + name

    if not os.path.exists(rcm_path):
        os.mkdir(rcm_path)
        print("RCM folder for %s data created" % rcm_path)
    else:
        print("RCM folder for %s data already exists" % rcm_path)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
        print("Output folder for %s data created" % out_path)
    else:
        print("Output folder for %s data already exists" % out_path)

#######################################################################################################
#
#   Function to write eight (8) different image files representing the different RCM kernel output
#   for a specfic image
#
#   Parameters:
#       name    file name of image (without extension)
#       abs     absolute gradients of RCM kernel output
#
#######################################################################################################
def generateRcmOutputs(name, grads, exts=RCM_EXTS):

    # !! CONFIIGURE FILE TYPE HERE !!
    file_type = ".png"
    file_path = "compasses\\" + name

    # Saving numpy file
    np_path = os.path.join(file_path, name + "_rcm_outputs.npy")
    np.save(np_path, grads)
    print("Generated array file: %s" % np_path)
    
    for i in range(len(grads)):
       img_path = os.path.join(file_path, name + "_" + exts[i] + file_type)
       cv.imwrite(img_path, grads[i])
       print("Generated image: %s" % img_path)

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

    filtered_images = [cv.filter2D(img, -1, mask) for mask in RCM]
    gradients = [np.abs(gradient) for gradient in filtered_images]

    # Generate image files of RCM kernel output
    generateRcmOutputs(name, gradients)

    return gradients
    
#######################################################################################################
#
#   The main function...to generate the final edge-detected image data
#
#   Parameters:
#       img     image data
#
#   Returns:
#       Post-processed image data
#
#######################################################################################################
def detectEdge(img, name):

    gradients = applyMasks(img, name)
    img_edged = np.sum(gradients, axis = 0)

    # write image to final folder
    output_f = "output\\" + name

    # Save np
    np_path = os.path.join(output_f, name + "_final_output.npy")
    np.save(np_path, img_edged)
    print("Generated array file: %s" % np_path)

    # Save image
    file_type = ".png"
    img_path = os.path.join(output_f, name + "_final_output" + file_type)
    cv.imwrite(img_path, img_edged.astype(np.uint8))
    print("Generated image: %s" % img_path)

    return img_edged 

#######################################################################################################
#
#   The main function...to generate the final edge-detected image data
#
#   Parameters:
#       img     image data
#
#   Returns:
#       Post-processed image data
#
#######################################################################################################
def detectEdgeAbs(img, name):

    gradients = applyMasks(img, name)
    img_edged = np.sum(gradients, axis = 0)

    # write image to final folder
    output_f = "output\\" + name

    # Save np
    np_path = os.path.join(output_f, name + "_final_output.npy")
    np.save(np_path, img_edged)
    print("Generated array file: %s" % np_path)

    # Save image
    file_type = ".png"
    img_path = os.path.join(output_f, name + "_final_output" + file_type)
    cv.imwrite(img_path, img_edged.astype(np.uint8))
    print("Generated image: %s" % img_path)

    return img_edged

#######################################################################################################
#
#   Another version of the main function...to generate the final edge-detected image data
#
#   Parameters:
#       img     image data
#       name    file name of image (without extension)
#
#   Returns:
#       Post-processed image data
#
#######################################################################################################
def detectEdgeMax(img, name, RCM=RCM_KERNELS):

    img_edged = np.zeros_like(img)
    for mask in RCM:
      filtered = cv.filter2D(img, -1, mask)
      # Perform edge detection operation enhancement
      # by combining the best resulting filtered_images into one
      np.maximum(img_edged, filtered, img_edged)

    # write image to final folder
    output_f = "output\\" + name

    # Save np
    np_path = os.path.join(output_f, name + "_final_output.npy")
    np.save(np_path, img_edged)
    print("Generated array file: %s" % np_path)

    # Save image
    file_type = ".png"
    img_path = os.path.join(output_f, name + "_final_output" + file_type)
    cv.imwrite(img_path, img_edged)
    print("Generated image: %s" % img_path)

    return img_edged
