# import the necessary packages
import imutils
import cv2
import numpy as np
def gradiente(img):

    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    angle = angle / (2 * 3.1415912)
    #print "valor maximo angle ", np.amax(angle)
    return angle

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			# print x, y
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def union(a,b, intersectionArea):
	(xA1, yA1, xA2, yA2) = a
	(xB1, yB1, xB2, yB2) = b
	w1 = max(xA1, xA2) - min(xA1, xA2)
	h1 = max(yA1, yA2) - min(yA1, yA2)
	w2 = max(xB1, xB2) - min(xB1, xB2)
	h2 = max(yB1, yB2) - min(yB1, yB2)
	return w1*h1 + w2*h2 - intersectionArea

def intersection(a, b):
	(xA1, yA1, xA2, yA2) = a
	(xB1, yB1, xB2, yB2) = b
	return max(0, min(xA2, xB2) - max(xA1, xB1)) * max(0, min(yA2, yB2) - max(yA1, yB1))

def IoU(gt, pr):
	interseccion = float(intersection(gt, pr))
	if round(interseccion / float(union(gt, pr, interseccion)), 3) == 0:
		return 0
	else:
		return round(interseccion / float(union(gt, pr, interseccion)), 3)
