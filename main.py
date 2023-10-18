import cv2
import numpy as np
import tensorflow as tf
camera = cv2.VideoCapture(0)
mymodel = tf.keras.models.load_model("keras_model.h5")
while True:
    status,frame = camera.read()
    if status:
        frame = cv2.flip(frame, 1)
        resize = cv2.resize(frame, (224,224))
        frame = np.expand_dims(resize,  x=0)
        frame = frame/255
        prediction = mymodel.predict(resize)
        rock = int(prediction[0][0]*100)
        paper = int(prediction[0][1]*100)
        scissor = int(prediction[0][2]*100)
        print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")
        cv2.imshow("feed", frame)
        cv2.waitKey(0)

camera.release()
cv2.destroyAllWindows()
    
    
    