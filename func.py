from audio import speechEmotionRecognition
import cv2
from moviepy.editor import VideoFileClip
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
all_pred=[]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
def extract_audio(video_path, audio_path='./extracted_audio/a.mp3'):
    """
    Extract audio from a video file and save it.

    Parameters:
    video_path (str): Path to the input video file.
    audio_path (str): Path to save the extracted audio file.
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    audio.close()
    video.close()
    print(f"Audio extracted and saved to {audio_path}")
def detect_emotion_func(video_path):
    model.load_weights('./models/model.h5')
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    cap = cv2.VideoCapture(video_path)  # Use video file
    extract_audio(video_path)
    ser=speechEmotionRecognition()
    predictions, timestamps = ser.predict_emotion_from_file('./extracted_audio/a.mp3')
    final_audio_emotion=np.mean(predictions, axis=0)
    print(f"The final audio emotion is: {emotion_dict[int(np.argmax(final_audio_emotion))]}")
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            all_pred.append(prediction)
            maxindex = int(np.argmax(prediction))
            # cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            print(emotion_dict[maxindex])
        # cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()
    # print(all_pred)
    
    avg_pred = np.mean(all_pred, axis=0)
    # avg_pred = (avg_pred + final_audio_emotion)/2
    final_emotion = emotion_dict[int(np.argmax(avg_pred))]
    print(f"The final emotion is: {final_emotion}")
    return final_emotion