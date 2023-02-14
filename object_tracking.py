# coding=utf-8 
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

(major, minor) = cv2.__version__.split(".")[:2]
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())
else:
	# opencv内嵌的tracking算法list
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.legacy.TrackerCSRT_create,
		"kcf": cv2.legacy.TrackerKCF_create,
		"boosting": cv2.legacy.TrackerBoosting_create,
		"mil": cv2.legacy.TrackerMIL_create,
		"tld": cv2.legacy.TrackerTLD_create,
		"medianflow": cv2.legacy.TrackerMedianFlow_create,
		"mosse": cv2.legacy.TrackerMOSSE_create
	}
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# init tracking object box
initBB = None
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
else:
	vs = cv2.VideoCapture(args["video"])
fps = None

while True:
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	if frame is None:
		break
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]

	if initBB:
		(success, box) = tracker.update(frame)
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
		fps.update()
		fps.stop()
		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("s"):
		# 按住's'键, 选择我们要跟踪的目标,
		# 也可直接输入box的h,w,width,height信息.
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
		tracker.init(frame, initBB)
		fps = FPS().start()
	elif key == ord("q"):
		# 退出tracking程序
		break

if not args.get("video", False):
	vs.stop()
else:
	vs.release()
cv2.destroyAllWindows()