from tkinter import Tk, Button, Label, filedialog
import cv2 # pip install opencv-contrib-python
from PIL import Image, ImageTk #pip install Pillow
import imutils
import mediapipe as mp
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import os
import numpy as np
import json

cap=None 
# Mediapipe #################################################################
mp_face_mesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils

# Read parameters ###########################################################
with open("parameters.json") as data:
    parameters=json.loads(data.read())

#Cargamos el modelo y pesos para la predicción
folder='resources/files/'
cnn = load_model(folder+'model.h5')
cnn.load_weights(folder+'weights.h5')
dire_img = os.listdir(folder+'validation')

def chooseVideo():
    global cap
    pathVideo=filedialog.askopenfilename(filetypes=[
        ("all video format", ".mp4")
    ])
    if len(pathVideo)>0:
        videoPathL['text']='Ruta del video: {}'.format(pathVideo)
        cap=cv2.VideoCapture(pathVideo)
        showVideo()

def streamingVideo():
    global cap
    cap=cv2.VideoCapture(0)
    showVideo()

def showVideo():
    global cap
    ret, frame=cap.read()
    # Getting backgroun image
    background=cv2.imread('resources/images/background.png')
    if ret == True:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ##############################################
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            min_detection_confidence=0.5
        ) as face_mesh:

            result=face_mesh.process(frame)
            detection=result.multi_face_landmarks

            if detection is not None:
                for face_landmarks in detection:
                    # Paintng dots and linies
                    mp_drawing.draw_landmarks(
                        background, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                        # Changing color
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=0, circle_radius=0),
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=4)
                    )
                    mp_drawing.draw_landmarks(
                        frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                        # Changing color
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=0, circle_radius=0),
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=4)
                    )

                x1, y1=220, 140 
                x2, y2=420, 340 
                cut=background[y1:y2, x1:x2]
                cut=cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
                cut=cv2.resize(cut, (parameters['width'], parameters['height']))
                vector = cnn.predict(np.expand_dims(img_to_array(cut), axis=0))
                r=vector[0]
                state=dire_img[np.argmax(vector[0])]
                faceContL['text']='Estado de mi boca: {}'.format(state)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 255, 255), 3)
        ##############################################


        i=Image.fromarray(frame)
        img=ImageTk.PhotoImage(image=i)
        videoL.configure(image=img)  
        videoL.image=img
        videoL.after(10, showVideo) 

window=Tk()
window.title('Detector de rostros')
window.state('zoomed')
window.configure(bg='#3e78dd')

# Buton para elegir video
seeB=Button(window, text='Elegir video', font=('Times', 24), command=chooseVideo)
seeB.grid(row=0, column=0, padx=10, pady=10)

# Buton para elegir video
seeStreamingB=Button(window, text='Usar cámara', font=('Times', 24), command=streamingVideo)
seeStreamingB.grid(row=0, column=1, padx=100, pady=10)

#Label para mostrar ruta
videoPathL=Label(window, text='Ruta del video: No hay algún video aún')
videoPathL.grid(row=1, column=0)
videoPathL.configure(bg='#3e78dd', font=('Times', 24))

# Label para el video
videoL=Label(window)
videoL.grid(row=3, column=0, padx=10)

#Contrador de rostros
faceContL=Label(window, text='Estado de mi boca: ', font=('Times', 24))
faceContL.grid(row=3, column=1, padx=10)

window.mainloop()