from collections import deque
from imutils.video import VideoStream
import numpy
import argparse
import cv2
import imutils


# determines the colour ball that will be tracked
while True:

    colour = input("What colour ball would you like to track? (Red, Orange, Green, Blue): ")

    if colour.lower() == "o" or colour.lower() == "orange":
        hsvLower = (2, 180, 153)
        hsvUpper = (21, 255, 255)
        break
    elif colour.lower() == "g" or colour.lower() == "green":
        hsvLower = (25, 82, 42)
        hsvUpper = (83, 255, 255)
        break
    elif colour.lower() == "r" or colour.lower() == "red":
        hsvLower = (119, 153, 0)
        hsvUpper = (179, 255, 255)
        break
    elif colour.lower() == 'b' or colour.lower() == "blue":
        hsvLower = (82, 68, 7)
        hsvUpper = (124, 255, 255)
        break

# initializes the argument parse and parses the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="Optional but if supplied, the path to the video file.")
parser.add_argument("-b", "--buffer", type=int, default=264, help="This is the max buffer size")
args = vars(parser.parse_args())

# initializes the list of tracked points and the max size it can become
coordinates = deque(maxlen=args["buffer"])

# if the video file is not supplied, access the webcam
# if the video file is supplied, grab a reference to the video file
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

# a loop that tracks the coloured ball
# stops if the user enters the q key or the video file runs out of frames
while True:
    # gets the current frame
    cur_frame = vs.read()

    # gets the frame from VideoCapture/VideoStream
    # frame = frame[1] if args.get("video", False) else frame
    if args.get("video", False):
        cur_frame = cur_frame[1]
    else:
        cur_frame

    # occurs when there is no frame, meaning the video file has run out
    # of frames so we break
    if cur_frame is None:
        break

    # resize the video frame and blurs it to reduce high frequency noise
    cur_frame = imutils.resize(cur_frame, width=900)
    blurred = cv2.GaussianBlur(cur_frame, (11, 11), 0)

    # converts BGR to the HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # creates a mask for the color within the HSV ranges
    mask = cv2.inRange(hsv, hsvLower, hsvUpper)

    # executes dilations and erosions to remove small blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # gets the contours in the mask, (the boundary of the colour)
    contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)

    # initializes the center of the ball as none
    center = None

    # ensures that at least one contour was found
    if len(contour) > 0:
        # find the largest contour in the mask and then computes the minimum
        # enclosing circle to find the centroid
        c = max(contour, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # finds the centroid (x, y)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # ensures that the radius of the circle calculated above is large enough
        if radius > 5:
            # draws the border around the ball using the coordinates of the boundaries
            # of the ball
            cv2.circle(cur_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # draws the centroid on the frame in red due to the BGR specified
            # and to a certain thickness
            cv2.circle(cur_frame, center, 5, (0, 0, 255), -1)

    # updates the coordinates queue of where the ball has been
    coordinates.appendleft(center)

    # creates the trail
    for i in range(1, len(coordinates)):
        # if the current coordinate or past coordinate was not detected,
        # we ignore the current index and continue looping
        if coordinates[i - 1] is None or coordinates[i] is None:
            continue

        # if both these points were successfully detected, create the desired
        # thickness of the line and connect the two points
        thickness = int(numpy.sqrt(args["buffer"] / (i+1)) * 2)
        cv2.line(cur_frame, coordinates[i - 1], coordinates[i], (0, 0, 255), thickness)

    # allows the frame to be visible on our screen
    cv2.imshow("Frame", cur_frame)

    # if q is entered by the user, the program terminates
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# if we are not using a video file, stop the camera stream
# if we are using a video file, release the camera
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

# close the window
cv2.destroyAllWindows()
