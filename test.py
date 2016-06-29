#!/usr/bin/env python

from CamCompass import CamCompass
from math import pi
import cv2

# Create object and register images
cc = CamCompass()
cc.registerFolder('./testimgs/angle90', pi/2)
cc.registerFolder('./testimgs/angle0', 0)
# After adding all images you can
# use cc.saveToFile(filename) and
# cc.readFromFile to avoid having
# to compute all SIFTs every time
# you run the program.

# This below is how you use it
# to query for an angle
img = cv2.imread('./testimgs/testimg.png')
angle = cc.getAngle(img)
print angle * 180.0/pi,"degrees"

