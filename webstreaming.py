# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
from pyimagesearch.image_captioning import ImageCaptioning
from pickle import TRUE, load
from flask_sock import Sock

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()
outputCaption = ""

# initialize a flask object
app = Flask(__name__)
sock = Sock(app)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src="http://81.83.10.9:8001/mjpg/video.mjpg").start()
#vs = VideoStream(src="http://81.149.56.38:8084/mjpg/video.mjpg").start()
vs = VideoStream(src="http://81.149.56.38:8081/mjpg/video.mjpg").start()

time.sleep(2.0)

description=[]
description.append("This page show a video")
description.append("a video stream at right")
description.append("This page show the description in the bottom")
description.append("the graph is at left")
description.append("The description start after this sentence")
description.append("")
description.append("")
description.append("")
description.append("")
description.append("")

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html",data=description)

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
		if total > frameCount:
			# detect motion in the image
			motion = md.detect(gray)

			# cehck to see if motion was found in the frame
			if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),
					(0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
		md.update(gray)
		total += 1

		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()

def captioning(framecount):

	global vs, outputFrame, lock, outputCaption
	cap = ImageCaptioning(model='New_Model.h5',tokenizer='New_Tok.pkl')
	n=-1

	while True:
		frame = vs.read()
		picture = imutils.resize(frame,width=400)
		caption = cap.run(frame)
		print(caption)
		n+=1
		if n==9:
			description[n]=caption
			n=(-1)
		if n<9:
			description[n]=caption

		""" descriptioncleant2 =[]
		descriptioncleant2.append(description[len(description)-10][9:(len(description)-7)])
		descriptioncleant2.append(description[len(description)-9][9:(len(description)-7)])
		descriptioncleant2.append(description[len(description)-8][9:(len(description)-7)])
		descriptioncleant2.append(description[len(description)-7][9:(len(description)-7)])
		descriptioncleant2.append(description[len(description)-6][9:(len(description)-7)])
		descriptioncleant2.append(description[len(description)-5][9:(len(description)-7)])
		descriptioncleant2.append(description[len(description)-4][9:(len(description)-7)])
		descriptioncleant2.append(description[len(description)-3][9:(len(description)-7)])
		descriptioncleant2.append(description[len(description)-2][9:(len(description)-7)])
		descriptioncleant2.append(description[len(description)-1][9:(len(description)-7)]) """

		
		
		
		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		cv2.putText(picture, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, picture.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		with lock:
			outputFrame = picture.copy()
			outputCaption = caption


def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@sock.route('/echo')
def echo(sock):
	global vs, outputFrame, lock, outputCaption
	while True:
		with lock:
			sock.send(outputCaption)
	

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str,
		help="ip address of the device",default="127.0.0.1") 
	ap.add_argument("-o", "--port", type=int,
		help="ephemeral port number of the server (1024 to 65535)",default=8000)
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=captioning, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	""" t2 = threading.Thread(target=echo,args=sock)
	t2.daemon = True
	t2.start() """

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()