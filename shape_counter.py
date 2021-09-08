import argparse, imutils, cv2

#construct the argument parser
ap = argparse.ArgumentParser()
#add the input command line argument
ap.add_argument("-i", "--input", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=True, help="path to output image")
args = vars(ap.parse_args())

#load the image from the given input from the user
#only loads and stores in a variable if the command line argument is given
image = cv2.imread(args["input"])

#convert the image to grayscale, blur it, and threshold it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

#extract contours from the image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#loop over the contours and draw them on the input image
for c in cnts:
    cv2.drawContours(image, [c], -1, (0,0,255), 2)

#display the total number of shapes on the image
text = "I found {} total shapes".format(len(cnts))
cv2.putText(image, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#write the output image to disk as specified by the cmd line args
cv2.imwrite(args["output"], image)
