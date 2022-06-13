from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tkinter as tk
pencere=tk.Tk()
pencere.geometry("400x400+50+100")
face_classifier = cv2.CascadeClassifier('../deneme/haarcascade_frontalface_default.xml')
classifier =load_model('../deneme/model3-2.h5')

emotion_labels = ['anger','contempt','disgust','fear','happy','neutral','sadness','surprise']
def Ac():
    cap = cv2.VideoCapture(0)

    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_classifier.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=8)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = classifier.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()
dugme=tk.Button(pencere,text="Kamera Ac", width=12, command=Ac).place(x=100,y=150)#ilk dugmenin ozellikleri place nerede konumlanacagını belirtiyor
dugme2=tk.Button(pencere,text="Pencereyi Kapat",command=pencere.quit).place(x=250,y=150)

pencere.mainloop()

