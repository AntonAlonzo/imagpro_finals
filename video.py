# Plate Number Recognition System using Robinsons Compass Mask Edge Detection | RCM Header File
# IMAGPRO | Alonzo, Hernandez, Solis, Susada

import os
import numpy as np
import cv2 as cv

while (True):
    ret, frame = vid.read()
    
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
vid.release()
cv.destroyAllWindows