import face_mask_detect_image as fmd
import argparse
import cv2
import time
import imutils
from imutils.video import VideoStream

class Video_Mask_Detector(fmd.Image_Mask_Detector):
    def __init__(self, face_model, mask_model, confidence, id, width, height):
        super().__init__(face_model, mask_model, confidence)
        self.camera_id = id
        self.width = width
        self.height = height
    
    def video_detect_mask(self, video):
        if not video == None:
            cap = cv2.VideoCapture(video)
        else:
            cap = cv2.VideoCapture(self.camera_id)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        time.sleep(2.0)
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            self.image_detect_mask(frame)

            cv2.imshow('frame',self.image_output)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video; if using webcam, leave it empty")
    ap.add_argument("-i", "--id", type=int, default=0, help="camera id")
    ap.add_argument("-f", "--face", type=str, default="faceDetector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="face_mask.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-wi", "--width", type=int, default=640, help="video's width")
    ap.add_argument("-hi", "--height", type=int, default=480, help="video's height")
    args = vars(ap.parse_args())

    mask_detector = Video_Mask_Detector(args["face"], args["model"], args["confidence"], args["id"], args["width"], args["height"])
    mask_detector.video_detect_mask(args["video"])