import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import cv2
import argparse

class Image_Mask_Detector:
    def __init__(self, face_model, mask_model, confidence):
        self.confidence = confidence
        # load face detector modal
        prototxtPath = os.path.sep.join([face_model, "deploy.prototxt"])
        weightsPath = os.path.sep.join([face_model, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.net = cv2.dnn.readNet(prototxtPath, weightsPath)
        # load face mask model
        self.model = keras.models.load_model(mask_model)

    def image_detect_mask(self, img):
        if not isinstance(img, np.ndarray):
            #load image
            img = cv2.imread(img)
        (height, width) = img.shape[:2]
        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence: #if detect a face
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall in the dimensions of the image
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

                #extract face
                face = img[startY:endY, startX:endX]
                if face.any():
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = keras.preprocessing.image.img_to_array(face)

                    #detect face mask
                    face = np.expand_dims(face, axis=0)
                    class_names = ["with mask", "without mask"]
                    predictions = self.model.predict(face)

                    score = tf.nn.softmax(predictions[0])

                    color = (0, 255, 0) if np.argmax(score) == 0 else (0, 0, 255)
                    label = "{} with a {:.2f} % confidence".format(class_names[np.argmax(score)], 100 * np.max(score))
                    cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        self.image_output = img

    def show_result(self):
        # show the output image
        cv2.imshow("Output", self.image_output)
        cv2.waitKey(0)

if __name__ == "__main__":
    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-f", "--face", type=str, default="faceDetector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="face_mask.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    mask_detector = Image_Mask_Detector(args["face"], args["model"], args["confidence"])
    mask_detector.image_detect_mask(args["image"])
    mask_detector.show_result()
