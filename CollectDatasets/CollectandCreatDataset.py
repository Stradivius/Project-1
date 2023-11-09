import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5
dataset_size = 100


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)



def CollectImage():
    global number_of_classes, dataset_size, DATA_DIR
    cap = cv2.VideoCapture(0)

    for i in range(number_of_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(i))):
            os.makedirs(os.path.join(DATA_DIR, str(i)))

        print(f"Collecting data for class {i}")

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, 'Press S to start', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('s'):
                break
            
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(i), f"{counter}.jpg"), frame)

            counter += 1


def CreateDatasets():
    global DATA_DIR, hands
    data = []
    labels = []
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

    with open("data.pickle", "wb") as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    

def train_classifier():
    data_dict = pickle.load(open('./data.pickle', 'rb'))

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_predict, y_test)

    print('{}% of samples were classified correctly !'.format(score * 100))

    with open("model.p", "wb") as f:
        pickle.dump({'model': model}, f)


def Rename():
    pass


root = tk.Tk()
root.title("Collect and Create Datasets")
root.iconbitmap("Data.ico")
root.geometry("500x552")
root.resizable(False, False)


img = Image.open("Dataimg.jpg")
img = ImageTk.PhotoImage(img)
label = Label(root, image=img)
label.place(x=0, y=0)


cambutton = Button(root,text="Access Webcam", font=("arial bold", 18), fg="white", bg="black", activebackground="white",activeforeground="black", command=CollectImage)
cambutton.place(x=150, y=200)


renamebutton = Button(root, text="Rename", font=("arial bold", 18), fg="white", bg="black", activebackground="white",activeforeground="black", command=Rename)
cambutton.place(x=150, y=260)


root.mainloop()