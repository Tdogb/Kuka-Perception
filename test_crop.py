import math

img_width = 640

cameraHeight = 1.27
cameraFOV = 2*math.pi * 110/360 #returns 0
realImageWidth = 2*(math.tan(cameraFOV/2)*cameraHeight)
widthRatio = realImageWidth / img_width

print(widthRatio)