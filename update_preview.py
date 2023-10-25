import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
import asyncio
from bleak import BleakClient, BleakScanner
import cv2
import mediapipe as mp
import math
from keras.src.saving.saving_api import load_model
import numpy as np
import os
import turtle
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)
hands_videos = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

""" t = turtle.Turtle() """
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)


send_abs = ""
tup = ()
devices_names = []
address = ""
characteristic = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"


def DetectandDrawHandsLandmarks(image, hands):
    output_image = image.copy()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    process_result = hands.process(imgRGB)
    if process_result.multi_hand_landmarks:
        for hand_landmarks in process_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=output_image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
            )
    return output_image, process_result


def countFingers(image, process_result):
    output_image = image.copy()
    count = {"RIGHT": 0}
    count_thumb_down = {"RIGHT": 0}
    count_thumb_up = {"RIGHT": 0}
    point_straight = {"RIGHT": 0}
    point_left = {"RIGHT": 0}
    point_right = {"RIGHT": 0}
    fingers_tips_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    fingers_statuses = {
        "THUMB": False,
        "INDEX": False,
        "MIDDLE": False,
        "RING": False,
        "PINKY": False,
    }
    for hand_index, hand_info in enumerate(process_result.multi_handedness):
        hand_label = "RIGHT"
        hand_landmarks = process_result.multi_hand_landmarks[hand_index]
        thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].y
        index_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        index_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
        for tip_index in fingers_tips_ids:
            finger_name = tip_index.name.split("_")[0]
            if (
                (hand_landmarks.landmark[tip_index - 3].y
                - hand_landmarks.landmark[tip_index].y) > 0.02 
            ):
                fingers_statuses[finger_name] = True
                count[hand_label] += 1
        if (
            (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP - 3].y
            - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y) > 0.06 and count[hand_label] == 1
        ):
            fingers_statuses["INDEX"] = True
            point_straight[hand_label] += 1
        if hand_label == "RIGHT" and (thumb_mcp_x - thumb_tip_x > 0.052):
            fingers_statuses["THUMB"] = True
            count[hand_label] += 1
        if hand_label == "RIGHT" and (thumb_mcp_y - thumb_tip_y > 0.09) and count[hand_label] == 0:
            fingers_statuses["THUMB"] = True
            count_thumb_up[hand_label] += 1
        elif hand_label == "RIGHT" and (thumb_tip_y - thumb_mcp_y > 0.09) and count[hand_label] == 0:
            fingers_statuses["THUMB"] = True
            count_thumb_down[hand_label] += 1
        if hand_label == "RIGHT" and (index_mcp_x - index_tip_x > 0.07) and count[hand_label] == 1:
            fingers_statuses["INDEX"] = True
            point_left[hand_label] += 1
        elif hand_label == "RIGHT" and (index_tip_x - index_mcp_x > 0.07) and count[hand_label] == 1:
            fingers_statuses["INDEX"] = True
            point_right[hand_label] += 1
    return output_image, fingers_statuses, count, count_thumb_up, count_thumb_down, point_left, point_right, point_straight 


def predictGestures1(image, fingers_statuses, count, count_thumb_up, count_thumb_down, process_result, point_left, point_right, point_straight, draw=True):
    output_image = image.copy()
    hand_label = "RIGHT"
    hands_gestures = {"RIGHT": "UNKNOWN"}
    color = (0, 0, 255)
    ang = getDegress(process_result)
    send_lst = [] 
    if (
        count_thumb_up[hand_label] == 1
        and fingers_statuses["THUMB"]
    ):
        hands_gestures[hand_label] = "LIKE"
        color = (0, 255, 0)
        send = "V"
        send_lst = []
        send_lst.append(send) 
    elif (
        count_thumb_down[hand_label] == 1
        and fingers_statuses["THUMB"]
    ):
        hands_gestures[hand_label] = "DISLIKE"
        color = (0, 255, 0)
        send = "Z"
        send_lst = []
        send_lst.append(send)
    elif (
        count[hand_label] == 3
        and fingers_statuses["THUMB"]
        and fingers_statuses["INDEX"]
        and fingers_statuses["PINKY"]
    ):
        hands_gestures[hand_label] = "ROCK"
        color = (0, 255, 0)
        send = "Y"
        send_lst = []
        send_lst.append(send)
    elif (
        point_left[hand_label] == 1
        and fingers_statuses["INDEX"] 
    ):
        hands_gestures[hand_label] = "POINT LEFT"
        color = (0, 255, 0)
        send = "L"
        send_lst = []
        send_lst.append(send)
    elif (
        point_right[hand_label] == 1
        and fingers_statuses["INDEX"]
    ):
        hands_gestures[hand_label] = "POINT RIGHT"
        color = (0, 255, 0)
        send = "R"
        send_lst = []
        send_lst.append(send)
    elif (
        point_straight[hand_label] == 1
        and fingers_statuses["INDEX"]
    ):
        hands_gestures[hand_label] = "POINT STRAIGHT"
        color = (0, 255, 0)
        send = "T"
        send_lst = []
        send_lst.append(send)
    if hands_gestures[hand_label] == "UNKNOWN":
        send = "N"
        send_lst = []
        send_lst.append(send)
    send_abs = ""
    send_abs = send_abs.join(send_lst)
    if draw:
        cv2.putText(
            output_image,
            f"{hands_gestures[hand_label]}: {ang}",
            (10, 100),
            cv2.FONT_HERSHEY_PLAIN,
            10,
            color,
            5,
        )
    return output_image, hands_gestures, send_abs


def predictGesturesD(image, fingers_statuses, count, process_result, draw=True):
    output_image = image.copy()
    hand_label = "RIGHT"
    hands_gestures = {"RIGHT": "UNKNOWN"}
    color = (0, 0, 255)
    ang = getDegress(process_result)
    send_lst = [] 
    ang_comparison = {(-60, -55): "A",
                      (-55, -50): "B",
                      (-50, -45): "C",
                      (-45, -40): "D",
                      (-40, -35): "E",
                      (-35, -30): "F",
                      (-30, -25): "G",
                      (-25, -20): "H",
                      (-20, -15): "I",
                      (-15, -10): "K",
                      (-10, -5): "L",
                      (-5, 0): "M",
                      (0, 5): "N",
                      (5, 10): "O",
                      (10, 15): "P",
                      (15, 20): "Q",
                      (20, 25): "V",
                      (25, 30): "S",
                      (30, 35): "U",
                      (35, 40): "R",
                      (40, 45): "X",
                      (45, 50): "Y",
                      (50, 55): "J",
                      (55, 60): "Z"
    }
    if (
        count[hand_label] == 2
        and fingers_statuses["MIDDLE"]
        and fingers_statuses["INDEX"]
    ):
        hands_gestures[hand_label] = "GUN SIGN"
        color = (0, 255, 0)
        for k in ang_comparison.keys(): 
            if k[0] < int(ang) < k[1]:
                send = ang_comparison[k]
                send_lst = []
                send_lst.append(send)
    if hands_gestures[hand_label] == "UNKNOWN":
        send = "1"
        send_lst = []
        send_lst.append(send)
    send_abs = ""
    send_abs = send_abs.join(send_lst)
    if draw:
        cv2.putText(
            output_image,
            f"{hands_gestures[hand_label]}: {ang}",
            (10, 100),
            cv2.FONT_HERSHEY_PLAIN,
            10,
            color,
            5,
            ) 
    return output_image, hands_gestures, send_abs


async def recognizeGesturesMode1():
    global camera_video, address, send_abs
    """ visual_robot("1") """
    """ async with BleakClient(address) as client:
        await client.write_gatt_char(data=bytes(send_abs, encoding="utf8"), char_specifier=characteristic, response=True)  """
    while camera_video.isOpened():
        ret, frame = camera_video.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame, process_result = DetectandDrawHandsLandmarks(frame, hands_videos)
        send_lst = []
        if process_result.multi_hand_landmarks:
            frame, finger_statuses, count, count_thumb_up, count_thumb_down, point_left, point_right, point_straight = countFingers(frame, process_result)
            frame, gestures, send = predictGestures1(frame, finger_statuses, count, count_thumb_up, count_thumb_down, process_result, point_left, point_right, point_straight)
            send_lst = []
            send_lst.append(send)
        cv2.imshow("Mode 1", frame)
        send_abs = ""
        send_abs = send_abs.join(send_lst) 
        print(send_abs)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyWindow("Mode 1")
            break
        if k == 50:
            cv2.destroyWindow("Mode 1")
            await recognizeGesturesMode2()
        if k == 68:
            cv2.destroyWindow("Mode 1")
            await recognizeGesturesModeD()


async def recognizeGesturesMode2():
    global hands, mp_hands, mp_drawing, address, camera_video
    model = load_model('mp_hand_gesture')
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)
    """ async with BleakClient(address) as client:
        await client.write_gatt_char(data=bytes(send_abs, encoding="utf8"), char_specifier=characteristic, response=True) """
    while True:
        _, frame = camera_video.read()
        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        frame1, _ = DetectandDrawHandsLandmarks(frame, hands_videos)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        className = ''
        send = []
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]
                send_lst_c = []
                classNamedict = {'okay':"O",
                                'peace': "P", 
                                'thumbs up': "U", 
                                'thumbs down': "D",
                                'call me': "C", 
                                'stop':"S", 
                                'rock':"Y",
                                'live long':"X", 
                                'fist':"J", 
                                'smile':"M"
                }
                if className in classNamedict:
                    send1 = classNamedict[className]
                    send_lst_c = []
                    send_lst_c.append(send1)
                send2 = ""
                send2 = send2.join(send_lst_c)
                send = []
                send.append(send2)
        cv2.putText(frame1, className.upper(), (10, 100), cv2.FONT_HERSHEY_PLAIN, 10, (0,255,0) ,5)
        cv2.imshow("Mode 2", frame1) 
        send_abs = ""
        send_abs = send_abs.join(send) 
        print(send_abs)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyWindow("Mode 2")
            break
        if k == 49:
            cv2.destroyWindow("Mode 2")
            await recognizeGesturesMode1()
        if k == 68:
            cv2.destroyWindow("Mode 2")
            await recognizeGesturesModeD()


async def recognizeGesturesModeD():
    global camera_video, address, send_abs
    """visual_robot("drive") """
    """async with BleakClient(address) as client:
        await client.write_gatt_char(data=bytes(send_abs, encoding="utf8"), char_specifier=characteristic, response=True)   """
    while camera_video.isOpened():
        ret, frame = camera_video.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame, process_result = DetectandDrawHandsLandmarks(frame, hands_videos)
        send_lst = []
        if process_result.multi_hand_landmarks:
            frame, finger_statuses, count, count_thumb_up, count_thumb_down, point_left, point_right, point_straight = countFingers(frame, process_result)
            frame, gestures, send = predictGesturesD(frame, finger_statuses, count, process_result)
            send_lst = []
            send_lst.append(send)
        cv2.imshow("Drive Mode", frame)
        send_abs = ""
        send_abs = send_abs.join(send_lst)
        print(send_abs)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyWindow("Drive Mode")
            break
        if k == 50:
            cv2.destroyWindow("Drive Mode")
            await recognizeGesturesMode2()
        if k == 49:
            cv2.destroyWindow("Drive Mode")
            await recognizeGesturesMode1()
            

def getDegress(process_result):
    anglelst = []
    for hand_landmark in process_result.multi_hand_landmarks:
        a1 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        a2 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        b1 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        b2 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        c1 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        c2 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y-0.05
        a = (a1,a2)
        b = (b1,b2)
        c = (c1,c2)
        ang = round(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))) 
        anglelst = []
        anglelst.append(str(ang))
    ang = " "
    ang = ang.join(anglelst)
    return ang


def first_click():
    frame1 = ttk.Frame(root, width=490, height=480)
    frame1.place(x=0, y=0)

    img = Image.open("KCbot.jpg")
    img = ImageTk.PhotoImage(img)
    label = Label(frame1, image=img)
    label.image = img
    label.place(x=0,y=0)

    option1 = Button(frame1, text="Mode 1", font=('arial bold', 18), fg="#000000", bg="#00b4d8", activebackground="#000000", activeforeground="#00b4d8", command=lambda: asyncio.run(recognizeGesturesMode1()))
    option1.place(x=180,y=80)

    option2 = Button(frame1, text="Mode 2", font=('arial bold', 18), fg="#000000", bg="#00b4d8", activebackground="#000000", activeforeground="#00b4d8", command=lambda: asyncio.run(recognizeGesturesMode2()))
    option2.place(x=180,y=200)

    option3 = Button(frame1, text="Drive Mode", font=('arial bold', 18), fg="#000000", bg="#00b4d8", activebackground="#000000", activeforeground="#00b4d8", command=lambda: asyncio.run(recognizeGesturesModeD()))
    option3.place(x=160,y=320)


def info():
    os.system('start Usage.docx')


async def Scan():
    global devices_names,tup
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)
        devices_names = []
        devices_names.append(d)
    with open(file="Scan Result.txt", mode='w') as file:
        file.truncate(0)
        for d in devices:
            file.writelines(d)
    tup = ()
    tup = tuple(devices_names)
    var1.set(1)
    label2 = Label(root, text=f"Found {len(devices_names)} devices", font=('arial bold', 8), fg="#000000", bg="#00b4d8")
    label2.place(x=180, y=160)
    osCommandString = "notepad.exe Scan_Result.txt"
    os.system(osCommandString)


def Connect():
    global address
    address = combobox.get()
    """if address == "":
        label2 = Label(root, text="Please choose a device!", font=('arial bold', 8), fg="#000000", bg="#00b4d8")
        label2.place(x=180, y=160)
    else: """
    address = address[0:17]
    var2.set(1)


""" def visual_robot(mode: str):
    turtle.getscreen()
    turtle.shape('square')
    turtle.pendown()
    turtle.color("blue")

    def changecolor():
        turtle.color("green")
        time.sleep(1)
        turtle.color("blue")
    
    if mode == "1":
        command = {"T": turtle.forward(2),
                    "L": turtle.left(10),
                    "R": turtle.right(10),
                    "Y": turtle.back(2),
                    "V": changecolor()}
    if send_abs in command.keys:
        command[send_abs] """
    

root = tk.Tk()
root.title("KC_bot controller by Hand Tracking")
root.geometry("480x470")
root.resizable(False, False)
root.iconbitmap("robot.ico")


img = Image.open("KCbot.jpg")
img = ImageTk.PhotoImage(img)
label1 = Label(root, image=img)
label1.place(x=0,y=0)

var1 = tk.IntVar()
var2 = tk.IntVar()
scanbutton_icon = tk.PhotoImage(file="bluetooth.png")
scanbutton_icon = scanbutton_icon.subsample(24,24)
scanbutton = Button(root, image=scanbutton_icon, command=lambda: asyncio.run(Scan()))
scanbutton.place(x=230, y=80)


scanbutton.wait_variable(var1)
connectbutton = Button(root, text="Connect", font=('arial bold', 8), fg="#000000", bg="#00b4d8", activebackground="#000000", activeforeground="#00b4d8", command=Connect)
connectbutton.place(x=290, y=120)


selected_device = tk.StringVar()
combobox = ttk.Combobox(root, textvariable=selected_device)
combobox['values']=tup
combobox['state']='readonly'
combobox.place(x=140,y=120)


connectbutton.wait_variable(var2)
webcambutton = Button(root, text="Access Webcam", font=('arial bold', 18), fg="#000000", bg="#00b4d8", activebackground="#000000", activeforeground="#00b4d8", command=first_click)
webcambutton.place(x=140,y=200)


infobutton = Button(root, text="How to use", font=('arial bold', 18), fg="#000000", bg="#00b4d8", activebackground="#000000", activeforeground="#00b4d8", command=info)
infobutton.place(x=165,y=320)


root.mainloop()