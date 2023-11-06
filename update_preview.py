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


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)
hands_videos = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)


send_abs = ""
tup = ()
devices_names = []
address = ""
characteristic = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"


thumb_distance = 5.0
index_distance = 5.0
middle_distance = 5.0
ring_distance = 5.0
pinky_distance = 5.0


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
    count = {"RIGHT": 0, "LEFT": 0}
    count_thumb_down = {"RIGHT": 0, "LEFT": 0}
    count_thumb_up = {"RIGHT": 0, "LEFT": 0}
    point_straight = {"RIGHT": 0, "LEFT": 0}
    point_left = {"RIGHT": 0, "LEFT": 0}
    point_right = {"RIGHT": 0, "LEFT": 0}
    fingers_tips_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    fingers_statuses = {
        "RIGHT_THUMB": False,
        "RIGHT_INDEX": False,
        "RIGHT_MIDDLE": False,
        "RIGHT_RING": False,
        "RIGHT_PINKY": False,
        "LEFT_THUMB": False,
        "LEFT_INDEX": False,
        "LEFT_MIDDLE": False,
        "LEFT_RING": False,
        "LEFT_PINKY": False,
    }
    for hand_index, hand_info in enumerate(process_result.multi_handedness):
        hand_label = str(hand_info.classification[0].label)
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
                hand_landmarks.landmark[tip_index - 3].y
                - hand_landmarks.landmark[tip_index].y
            ) > 0.02:
                fingers_statuses[hand_label.upper() + "_" + finger_name] = True
                count[hand_label.upper()] += 1
        if (
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP - 3].y
            - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        ) > 0.06 and count[hand_label.upper()] == 1:
            fingers_statuses[hand_label.upper() + "_INDEX"] = True
            point_straight[hand_label.upper()] += 1
        if (hand_label.upper() == "RIGHT" and (thumb_mcp_x - thumb_tip_x > 0.052)) or (
            hand_label.upper() == "LEFT" and (thumb_tip_x - thumb_mcp_x)
        ) > 0.052:
            fingers_statuses[hand_label.upper() + "_THUMB"] = True
            count[hand_label.upper()] += 1
        if (thumb_mcp_y - thumb_tip_y > 0.09) and count[hand_label.upper()] == 0:
            fingers_statuses[hand_label.upper() + "_THUMB"] = True
            count_thumb_up[hand_label.upper()] += 1
        elif (thumb_tip_y - thumb_mcp_y > 0.09) and count[hand_label.upper()] == 0:
            fingers_statuses[hand_label.upper() + "_THUMB"] = True
            count_thumb_down[hand_label.upper()] += 1
        if (
            hand_label.upper() == "RIGHT"
            and (index_mcp_x - index_tip_x > 0.07)
            and count[hand_label.upper()] == 1
        ) or (
            hand_label.upper() == "RIGHT"
            and (index_tip_x - index_mcp_x > 0.07)
            and count[hand_label.upper()] == 1
        ):
            fingers_statuses[hand_label.upper() + "_INDEX"] = True
            point_left[hand_label.upper()] += 1
        elif (
            hand_label.upper() == "RIGHT"
            and (index_tip_x - index_mcp_x > 0.07)
            and count[hand_label.upper()] == 1
        ) or (
            hand_label.upper() == "LEFT"
            and (index_mcp_x - index_tip_x > 0.07)
            and count[hand_label.upper()] == 1
        ):
            fingers_statuses[hand_label.upper() + "_INDEX"] = True
            point_right[hand_label.upper()] += 1
    return (
        output_image,
        fingers_statuses,
        count,
        count_thumb_up,
        count_thumb_down,
        point_left,
        point_right,
        point_straight,
    )


def predictGestures1(
    image,
    process_result,
    fingers_statuses,
    count,
    count_thumb_up,
    count_thumb_down,
    point_left,
    point_right,
    point_straight,
    draw=True,
):
    output_image = image.copy()
    hands_gestures = {"RIGHT": "UNKNOWN", "LEFT": "UNKNOWN"}
    for hand_index, hand_info in enumerate(process_result.multi_handedness):
        hand_label = str(hand_info.classification[0].label)
        color = (0, 0, 255)
        ang = getDegress(process_result)
        send_lst = []
        if (
            count_thumb_up[hand_label.upper()] == 1
            and fingers_statuses[hand_label.upper() + "_THUMB"]
        ):
            hands_gestures[hand_label.upper()] = "LIKE"
            color = (0, 255, 0)
            send = "V"
            send_lst = []
            send_lst.append(send)
        elif (
            count_thumb_down[hand_label.upper()] == 1
            and fingers_statuses[hand_label.upper() + "_THUMB"]
        ):
            hands_gestures[hand_label.upper()] = "DISLIKE"
            color = (0, 255, 0)
            send = "Z"
            send_lst = []
            send_lst.append(send)
        elif (
            count[hand_label.upper()] == 3
            and fingers_statuses[hand_label.upper() + "_THUMB"]
            and fingers_statuses[hand_label.upper() + "_INDEX"]
            and fingers_statuses[hand_label.upper() + "_PINKY"]
        ):
            hands_gestures[hand_label.upper()] = "ROCK"
            color = (0, 255, 0)
            send = "Y"
            send_lst = []
            send_lst.append(send)
        elif (
            point_left[hand_label.upper()] == 1
            and fingers_statuses[hand_label.upper() + "_INDEX"]
        ):
            hands_gestures[hand_label.upper()] = "POINT LEFT"
            color = (0, 255, 0)
            send = "L"
            send_lst = []
            send_lst.append(send)
        elif (
            point_right[hand_label.upper()] == 1
            and fingers_statuses[hand_label.upper() + "_INDEX"]
        ):
            hands_gestures[hand_label.upper()] = "POINT RIGHT"
            color = (0, 255, 0)
            send = "R"
            send_lst = []
            send_lst.append(send)
        elif (
            point_straight[hand_label.upper()] == 1
            and fingers_statuses[hand_label.upper() + "_INDEX"]
        ):
            hands_gestures[hand_label.upper()] = "POINT STRAIGHT"
            color = (0, 255, 0)
            send = "T"
            send_lst = []
            send_lst.append(send)
        if hands_gestures[hand_label.upper()] == "UNKNOWN":
            send = "N"
            send_lst = []
            send_lst.append(send)
        send_abs = ""
        send_abs = send_abs.join(send_lst)
        if draw:
            cv2.putText(
                output_image,
                f"{hands_gestures[hand_label.upper()]}: {ang}",
                (10, 100),
                cv2.FONT_HERSHEY_PLAIN,
                10,
                color,
                5,
            )
    return output_image, hands_gestures, send_abs


def predictGesturesD(image, process_result, fingers_statuses, count, draw=True):
    output_image = image.copy()
    hands_gestures = {"RIGHT": "UNKNOWN", "LEFT": "UNKNOWN"}
    color = (0, 0, 255)
    ang = getDegress(process_result)
    send_lst = []
    ang_comparison = {
        (-60, -55): "A",
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
        (55, 60): "Z",
    }
    for hand_index, hand_info in enumerate(process_result.multi_handedness):
        hand_label = str(hand_info.classification[0].label)
        if (
            count[hand_label.upper()] == 2
            and fingers_statuses[hand_label.upper() + "_MIDDLE"]
            and fingers_statuses[hand_label.upper() + "_INDEX"]
        ):
            hands_gestures[hand_label.upper()] = "GUN SIGN"
            color = (0, 255, 0)
            for k in ang_comparison.keys():
                if k[0] < int(ang) < k[1]:
                    send = ang_comparison[k]
                    send_lst = []
                    send_lst.append(send)
        if hands_gestures[hand_label.upper()] == "UNKNOWN":
            send = "1"
            send_lst = []
            send_lst.append(send)
        send_abs = ""
        send_abs = send_abs.join(send_lst)
        if draw:
            cv2.putText(
                output_image,
                f"{hands_gestures[hand_label.upper()]}: {ang}",
                (10, 100),
                cv2.FONT_HERSHEY_PLAIN,
                10,
                color,
                5,
            )
    return output_image, hands_gestures, send_abs


async def recognizeGesturesMode1D(mode):
    global camera_video, address, send_abs
    while camera_video.isOpened():
        ret, frame = camera_video.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame, process_result = DetectandDrawHandsLandmarks(frame, hands_videos)
        send_lst = []
        if process_result.multi_hand_landmarks:
            (
                frame,
                finger_statuses,
                count,
                count_thumb_up,
                count_thumb_down,
                point_left,
                point_right,
                point_straight,
            ) = countFingers(frame, process_result)
            if mode == "1":
                frame, gestures, send = predictGestures1(
                    frame,
                    process_result,
                    finger_statuses,
                    count,
                    count_thumb_up,
                    count_thumb_down,
                    point_left,
                    point_right,
                    point_straight,
                )
            elif mode == "D":
                frame, gestures, send = predictGesturesD(
                frame, process_result, finger_statuses, count
                )
            send_lst = []
            send_lst.append(send)
        if mode == "1":
            cv2.imshow("Mode 1", frame)
        elif mode == "D":
            cv2.imshow("Mode D", frame)
        send_abs = ""
        send_abs = send_abs.join(send_lst)
        print(send_abs)
        k = cv2.waitKey(1) & 0xFF
        if mode == "1": 
            if k == 27:
                cv2.destroyWindow("Mode 1")
                break
            if k == 50:
                cv2.destroyWindow("Mode 1")
                await recognizeGesturesMode2()
            if k == 68:
                cv2.destroyWindow("Mode 1")
                await recognizeGesturesMode1D(mode == "D")
        elif mode == "D": 
            if k == 27:
                cv2.destroyWindow("Mode D")
                break
            if k == 49:
                cv2.destroyWindow("Mode D")
                await recognizeGesturesMode1D(mode == "1")
            if k == 50:
                cv2.destroyWindow("Mode D")
                await recognizeGesturesMode2


async def recognizeGesturesMode2():
    global hands, mp_hands, mp_drawing, address, camera_video
    model = load_model("mp_hand_gesture")
    f = open("gesture.names", "r")
    classNames = f.read().split("\n")
    f.close()
    print(classNames)
    while True:
        _, frame = camera_video.read()
        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        frame1, _ = DetectandDrawHandsLandmarks(frame, hands_videos)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        className = ""
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
                classNamedict = {
                    "okay": "O",
                    "peace": "P",
                    "thumbs up": "U",
                    "thumbs down": "D",
                    "call me": "C",
                    "stop": "S",
                    "rock": "Y",
                    "live long": "X",
                    "fist": "J",
                    "smile": "M",
                }
                if className in classNamedict:
                    send1 = classNamedict[className]
                    send_lst_c = []
                    send_lst_c.append(send1)
                send2 = ""
                send2 = send2.join(send_lst_c)
                send = []
                send.append(send2)
        cv2.putText(
            frame1,
            className.upper(),
            (10, 100),
            cv2.FONT_HERSHEY_PLAIN,
            10,
            (0, 255, 0),
            5,
        )
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


def getDegress(process_result):
    anglelst = []
    for hand_landmark in process_result.multi_hand_landmarks:
        a1 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        a2 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        b1 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        b2 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        c1 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        c2 = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y - 0.05
        a = (a1, a2)
        b = (b1, b2)
        c = (c1, c2)
        ang = round(
            math.degrees(
                math.atan2(c[1] - b[1], c[0] - b[0])
                - math.atan2(a[1] - b[1], a[0] - b[0])
            )
        )
        anglelst = []
        anglelst.append(str(ang))
    ang = " "
    ang = ang.join(anglelst)
    return ang


def MeasureDistance(process_result):
    thumb_distance_lst = []
    index_distance_lst = []
    middle_distance_lst = []
    ring_distance_lst = []
    pinky_distance_lst = []
    for hand_index, _ in enumerate(process_result.multi_handedness):
        hand_landmark = process_result.multi_hand_landmarks[hand_index]
        thumb_tip_x = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_MCP].x
        index_tip_y = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        index_mcp_y = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        middle_tip_y = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        middle_mcp_y = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        ring_tip_y = hand_landmark.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_mcp_y = hand_landmark.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
        pinky_tip_y = hand_landmark.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_mcp_y = hand_landmark.landmark[mp_hands.HandLandmark.PINKY_MCP].y
        thumb_distance_lst.clear()
        index_distance_lst.clear()
        middle_distance_lst.clear()
        ring_distance_lst.clear()
        pinky_distance_lst.clear()
        thumb_distance_1 = str(abs(thumb_tip_x - thumb_mcp_x))
        index_distance_1 = str(abs(index_tip_y - index_mcp_y))
        middle_distance_1 = str(abs(middle_tip_y - middle_mcp_y))
        ring_distance_1 = str(abs(ring_tip_y - ring_mcp_y))
        pinky_distance_1 = str(abs(pinky_tip_y - pinky_mcp_y))
        thumb_distance_lst.append(thumb_distance_1)
        index_distance_lst.append(index_distance_1)
        middle_distance_lst.append(middle_distance_1)
        ring_distance_lst.append(ring_distance_1)
        pinky_distance_lst.append(pinky_distance_1)
    thumb_distance_2 = ""
    thumb_distance_2 = thumb_distance_2.join(thumb_distance_lst)
    index_distance_2 = ""
    index_distance_2 = index_distance_2.join(index_distance_lst)
    middle_distance_2 = ""
    middle_distance_2 = middle_distance_2.join(middle_distance_lst)
    ring_distance_2 = ""
    ring_distance_2 = ring_distance_2.join(ring_distance_lst)
    pinky_distance_2 = ""
    pinky_distance_2 = pinky_distance_2.join(pinky_distance_lst)  
    return (
        thumb_distance_2,
        index_distance_2,
        middle_distance_2,
        ring_distance_2,
        pinky_distance_2,
    )


def ShowVideoWhileMeasuring(mode):
    global thumb_distance, index_distance, middle_distance, ring_distance, pinky_distance
    if os.path.exists("Thumb.txt"):
        color_t = (0,255,0)
    else:
        color_t = (0,0,255)
    if os.path.exists("Index.txt"):
        color_i = (0,255,0)
    else:
        color_i = (0,0,255)
    if os.path.exists("Middle.txt"):
        color_m = (0,255,0)
    else:
        color_m = (0,0,255)
    if os.path.exists("Ring.txt"):
        color_r = (0,255,0)
    else:
        color_r = (0,0,255)
    if os.path.exists("Pinky.txt"):
        color_p = (0,255,0)
    else:
        color_p = (0,0,255)
    while camera_video.isOpened():
        ret, frame = camera_video.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame, process_result = DetectandDrawHandsLandmarks(frame, hands_videos)
        if process_result.multi_hand_landmarks:
            (
                thumb_distance_2,
                index_distance_2,
                middle_distance_2,
                ring_distance_2,
                pinky_distance_2,
            ) = MeasureDistance(process_result)
            thumb_distance = float(thumb_distance_2)
            index_distance = float(index_distance_2)
            middle_distance = float(middle_distance_2)
            ring_distance = float(ring_distance_2)
            pinky_distance = float(pinky_distance_2)
            if os.path.exists("Thumb.txt"):
                with open("Thumb.txt", "r") as f:
                    thumb_distance = float(f.read())
            if os.path.exists("Index.txt"):
                with open("Index.txt", "r") as f:
                    index_distance = float(f.read())
            if os.path.exists("Middle.txt"):
                with open("Middle.txt", "r") as f:
                    middle_distance = float(f.read())
            if os.path.exists("Ring.txt"):
                with open("Ring.txt", "r") as f:
                    ring_distance = float(f.read())
            if os.path.exists("Pinky.txt"):
                with open("Pinky.txt", "r") as f:
                    pinky_distance = float(f.read())
            cv2.putText(
                frame,
                f"THUMB: {thumb_distance}",
                (10, 100),
                cv2.FONT_HERSHEY_PLAIN,
                4,
                color_t,
                2,
            )
            cv2.putText(
                frame,
                f"INDEX: {index_distance}",
                (10, 150),
                cv2.FONT_HERSHEY_PLAIN,
                4,
                color_i,
                2,
            )
            cv2.putText(
                frame,
                f"MIDDLE: {middle_distance}",
                (10, 200),
                cv2.FONT_HERSHEY_PLAIN,
                4,
                color_m,
                2,
            )
            cv2.putText(
                frame,
                f"RING: {ring_distance}",
                (10, 250),
                cv2.FONT_HERSHEY_PLAIN,
                4,
                color_r,
                2,
            )
            cv2.putText(
                frame,
                f"PINKY: {pinky_distance}",
                (10, 300),
                cv2.FONT_HERSHEY_PLAIN,
                4,
                color_p,
                2,
            )
        cv2.imshow("Measuring", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 116:
            color_t = (0,255,0)
            with open("Thumb.txt", "w") as f:
                f.truncate(0)
                for i in range(1):
                    f.writelines(f"{thumb_distance_2}")
        elif k == 105:
            color_i = (0,255,0)
            with open("Index.txt", "w") as f:
                f.truncate(0)
                for i in range(1):
                    f.writelines(f"{index_distance_2}")
        elif k == 109:
            color_m = (0,255,0)
            with open("Middle.txt", "w") as f:
                f.truncate(0)
                for i in range(1):
                    f.writelines(f"{middle_distance_2}")
        elif k == 114:
            color_r = (0,255,0)
            with open("Ring.txt", "w") as f:
                f.truncate(0)
                for i in range(1):
                    f.writelines(f"{ring_distance_2}")
        elif k == 112:
            color_p = (0,255,0)
            with open("Pinky.txt", "w") as f:
                f.truncate(0)
                for i in range(1):
                    f.writelines(f"{pinky_distance_2}")
        elif k == 8:
            try: 
                os.remove("Thumb.txt")
                thumb_distance = float(thumb_distance_2)
                color_t = (0,0,255)
            except FileNotFoundError:
                pass
            try: 
                os.remove("Index.txt")
                index_distance = float(index_distance_2)
                color_i = (0,0,255)
            except FileNotFoundError:
                pass
            try: 
                os.remove("Ring.txt")
                ring_distance = float(ring_distance_2)
                color_m = (0,0,255)
            except FileNotFoundError:
                pass
            try: 
                os.remove("Middle.txt")
                middle_distance = float(middle_distance_2)
                color_r = (0,0,255)
            except FileNotFoundError:
                pass
            try: 
                os.remove("Pinky.txt")
                pinky_distance = float(pinky_distance_2)
                color_p = (0,0,255)
            except FileNotFoundError:
                pass
        else:
            pass


        if os.path.exists("Thumb.txt") and os.path.exists("Index.txt") and os.path.exists("Middle.txt") and os.path.exists("Ring.txt") and os.path.exists("Pinky.txt"):
            if k == 32 and mode == "1":
                asyncio.run(recognizeGesturesMode1D_with_Measurement(thumb_distance, index_distance, middle_distance, ring_distance, pinky_distance, mode = "1"))
            if k == 32 and mode == "D":
                asyncio.run(recognizeGesturesMode1D_with_Measurement(thumb_distance, index_distance, middle_distance, ring_distance, pinky_distance, mode = "D"))
        else:
            pass


def CustomcountFingers(image, process_result, thumb_distance, index_distance, middle_distance, ring_distance, pinky_distance):
    output_image = image.copy()
    count = {"RIGHT": 0, "LEFT": 0}
    count_thumb_down = {"RIGHT": 0, "LEFT": 0}
    count_thumb_up = {"RIGHT": 0, "LEFT": 0}
    point_straight = {"RIGHT": 0, "LEFT": 0}
    point_left = {"RIGHT": 0, "LEFT": 0}
    point_right = {"RIGHT": 0, "LEFT": 0}
    fingers_statuses = {
        "RIGHT_THUMB": False,
        "RIGHT_INDEX": False,
        "RIGHT_MIDDLE": False,
        "RIGHT_RING": False,
        "RIGHT_PINKY": False,
        "LEFT_THUMB": False,
        "LEFT_INDEX": False,
        "LEFT_MIDDLE": False,
        "LEFT_RING": False,
        "LEFT_PINKY": False,
    }
    for hand_index, hand_info in enumerate(process_result.multi_handedness):
        hand_label = str(hand_info.classification[0].label)
        hand_landmarks = process_result.multi_hand_landmarks[hand_index]
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
        thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
        index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        index_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        index_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        index_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        middle_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
        pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
        if (index_mcp_y - index_tip_y > index_distance):
            count[hand_label.upper()] += 1
            fingers_statuses[hand_label.upper() + "_INDEX"] = True
        if (middle_mcp_y - middle_tip_y > middle_distance):
            count[hand_label.upper()] += 1
            fingers_statuses[hand_label.upper() + "_MIDDLE"] = True
        if (ring_mcp_y - ring_tip_y > ring_distance):
            count[hand_label.upper()] += 1
            fingers_statuses[hand_label.upper() + "_RING"] = True
        if (pinky_mcp_y - pinky_tip_y > pinky_distance):
            count[hand_label.upper()] += 1
            fingers_statuses[hand_label.upper() + "_PINKY"] = True
        if (index_mcp_y - index_tip_y) > index_distance and count[hand_label.upper()] == 1:
            fingers_statuses[hand_label.upper() + "_INDEX"] = True
            point_straight[hand_label.upper()] += 1
        if (hand_label.upper() == "RIGHT" and (thumb_mcp_x - thumb_tip_x > thumb_distance)) or (
            hand_label.upper() == "LEFT" and (thumb_tip_x - thumb_mcp_x)
        ) > thumb_distance:
            fingers_statuses[hand_label.upper() + "_THUMB"] = True
            count[hand_label.upper()] += 1
        if (thumb_mcp_y - thumb_tip_y > thumb_distance) and count[hand_label.upper()] == 0:
            fingers_statuses[hand_label.upper() + "_THUMB"] = True
            count_thumb_up[hand_label.upper()] += 1
        elif (thumb_tip_y - thumb_mcp_y > thumb_distance) and count[hand_label.upper()] == 0:
            fingers_statuses[hand_label.upper() + "_THUMB"] = True
            count_thumb_down[hand_label.upper()] += 1
        if (
            hand_label.upper() == "RIGHT"
            and (index_mcp_x - index_tip_x > index_distance)
            and count[hand_label.upper()] == 1
        ) or (
            hand_label.upper() == "RIGHT"
            and (index_tip_x - index_mcp_x > index_distance)
            and count[hand_label.upper()] == 1
        ):
            fingers_statuses[hand_label.upper() + "_INDEX"] = True
            point_left[hand_label.upper()] += 1
        elif (
            hand_label.upper() == "RIGHT"
            and (index_tip_x - index_mcp_x > index_distance)
            and count[hand_label.upper()] == 1
        ) or (
            hand_label.upper() == "LEFT"
            and (index_mcp_x - index_tip_x > index_distance)
            and count[hand_label.upper()] == 1
        ):
            fingers_statuses[hand_label.upper() + "_INDEX"] = True
            point_right[hand_label.upper()] += 1
    return (
        output_image,
        fingers_statuses,
        count,
        count_thumb_up,
        count_thumb_down,
        point_left,
        point_right,
        point_straight,
    )


async def recognizeGesturesMode1D_with_Measurement(thumb_distance, index_distance, middle_distance, ring_distance, pinky_distance, mode): 
    global camera_video, address, send_abs
    while camera_video.isOpened():
        ret, frame = camera_video.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame, process_result = DetectandDrawHandsLandmarks(frame, hands_videos)
        send_lst = []
        if process_result.multi_hand_landmarks:
            (
                frame,
                finger_statuses,
                count,
                count_thumb_up,
                count_thumb_down,
                point_left,
                point_right,
                point_straight,
            ) = CustomcountFingers(image = frame, process_result=process_result, thumb_distance = thumb_distance, index_distance=index_distance,middle_distance = middle_distance, ring_distance=ring_distance, pinky_distance =pinky_distance)
            if mode == "1":
                frame, gestures, send = predictGestures1(
                    frame,
                    process_result,
                    finger_statuses,
                    count,
                    count_thumb_up,
                    count_thumb_down,
                    point_left,
                    point_right,
                    point_straight,
                )
            elif mode == "D": 
                frame, gestures, send = predictGesturesD(
                    frame,
                    process_result,
                    finger_statuses,
                    count
                )
            send_lst = []
            send_lst.append(send)
        if mode == "1":
            cv2.imshow("Mode 1 with Measurement", frame)
        elif mode == "D":
            cv2.imshow("Mode D with Measurement", frame)
        send_abs = ""
        send_abs = send_abs.join(send_lst)
        print(send_abs)
        k = cv2.waitKey(1) & 0xFF
        if mode == "1":
            if k == 27:
                cv2.destroyWindow("Mode 1 with Measurement")
                break
            if k == 50:
                cv2.destroyWindow("Mode 1 with Measurement")
                await recognizeGesturesMode2()
            if k == 68:
                cv2.destroyWindow("Mode 1 with Measurement")
                await recognizeGesturesMode1D(mode="D")
        elif mode == "D":
            if k == 27:
                cv2.destroyWindow("Mode D with Measurement")
                break
            if k == 50:
                cv2.destroyWindow("Mode D with Measurement")
                await recognizeGesturesMode2()
            if k == 49:
                cv2.destroyWindow("Mode D with Measurement")
                await recognizeGesturesMode1D(mode="1")


def first_click():
    frame1 = ttk.Frame(root, width=490, height=480)
    frame1.place(x=0, y=0)

    img = Image.open("KCbot.jpg")
    img = ImageTk.PhotoImage(img)
    label = Label(frame1, image=img)
    label.image = img
    label.place(x=0, y=0)

    option1 = Button(
        frame1,
        text="Mode 1",
        font=("arial bold", 18),
        fg="#000000",
        bg="#00b4d8",
        activebackground="#000000",
        activeforeground="#00b4d8",
        command=second_click,
    )
    option1.place(x=180, y=80)

    option2 = Button(
        frame1,
        text="Mode 2",
        font=("arial bold", 18),
        fg="#000000",
        bg="#00b4d8",
        activebackground="#000000",
        activeforeground="#00b4d8",
        command=lambda: asyncio.run(recognizeGesturesMode2()),
    )
    option2.place(x=180, y=200)

    option3 = Button(
        frame1,
        text="Drive Mode",
        font=("arial bold", 18),
        fg="#000000",
        bg="#00b4d8",
        activebackground="#000000",
        activeforeground="#00b4d8",
        command=third_click,
    )
    option3.place(x=160, y=320)


def info():
    os.system("start Usage.docx")


async def Scan():
    global devices_names, tup, combobox
    devices_names = []
    tup = ()
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)
        devices_names.append(d)
    tup = tuple(devices_names)
    var1.set(1)
    label2 = Label(
        root,
        text=f"Found {len(devices_names)} devices",
        font=("arial bold", 8),
        fg="#000000",
        bg="#00b4d8",
    )
    label2.place(x=180, y=160)
    selected_device = tk.StringVar()
    combobox = ttk.Combobox(root, textvariable=selected_device)
    combobox["values"] = tup
    combobox["state"] = "readonly"
    combobox.place(x=140, y=120)


def Connect():
    global address
    address = combobox.get()
    """ if address == "":
        label2 = Label(
            root,
            text="Please choose a device!",
            font=("arial bold", 8),
            fg="#000000",
            bg="#00b4d8",
        )
        label2.place(x=180, y=160)
    else: """
    address = address[0:17]
    var2.set(1)


def second_click():
    frame2 = ttk.Frame(root, width=490, height=480)
    frame2.place(x=0, y=0)

    img = Image.open("KCbot.jpg")
    img = ImageTk.PhotoImage(img)
    label = Label(frame2, image=img)
    label.image = img
    label.place(x=0, y=0)

    default2 = Button(
        frame2,
        text="Continue with default measurement",
        font=("arial bold", 18),
        fg="#000000",
        bg="#00b4d8",
        activebackground="#000000",
        activeforeground="#00b4d8",
        command=lambda: asyncio.run(recognizeGesturesMode1D(mode="1")),
    )
    default2.place(x=30, y=120)

    measure2 = Button(
        frame2,
        text="Customize your own measurement",
        font=("arial bold", 18),
        fg="#000000",
        bg="#00b4d8",
        activebackground="#000000",
        activeforeground="#00b4d8",
        command=lambda: ShowVideoWhileMeasuring(mode="1"),
    )
    measure2.place(x=35, y=280)


def third_click():
    frame2 = ttk.Frame(root, width=490, height=480)
    frame2.place(x=0, y=0)

    img = Image.open("KCbot.jpg")
    img = ImageTk.PhotoImage(img)
    label = Label(frame2, image=img)
    label.image = img
    label.place(x=0, y=0)

    default3 = Button(
        frame2,
        text="Continue with default measurement",
        font=("arial bold", 18),
        fg="#000000",
        bg="#00b4d8",
        activebackground="#000000",
        activeforeground="#00b4d8",
        command=lambda: asyncio.run(recognizeGesturesMode1D(mode="D")),
    )
    default3.place(x=30, y=120)

    measure3 = Button(
        frame2,
        text="Customize your own measurement",
        font=("arial bold", 18),
        fg="#000000",
        bg="#00b4d8",
        activebackground="#000000",
        activeforeground="#00b4d8",
        command=lambda: ShowVideoWhileMeasuring(mode="D"),
    )
    measure3.place(x=35, y=280)


root = tk.Tk()
root.title("KCBot controller by Hand Tracking")
root.geometry("480x470")
root.resizable(False, False)
root.iconbitmap("robot.ico")


img = Image.open("KCbot.jpg")
img = ImageTk.PhotoImage(img)
label1 = Label(root, image=img)
label1.place(x=0, y=0)


var1 = tk.IntVar()
var2 = tk.IntVar()
scanbutton_icon = tk.PhotoImage(file="bluetooth.png")
scanbutton_icon = scanbutton_icon.subsample(24, 24)
scanbutton = Button(root, image=scanbutton_icon, command=lambda: asyncio.run(Scan()))
scanbutton.place(x=230, y=80)


scanbutton.wait_variable(var1)
connectbutton = Button(
    root,
    text="Connect",
    font=("arial bold", 8),
    fg="#000000",
    bg="#00b4d8",
    activebackground="#000000",
    activeforeground="#00b4d8",
    command=Connect,
)
connectbutton.place(x=290, y=120)


connectbutton.wait_variable(var2)
webcambutton = Button(
    root,
    text="Access Webcam",
    font=("arial bold", 18),
    fg="#000000",
    bg="#00b4d8",
    activebackground="#000000",
    activeforeground="#00b4d8",
    command=first_click,
)
webcambutton.place(x=140, y=200)


infobutton = Button(
    root,
    text="How to use",
    font=("arial bold", 18),
    fg="#000000",
    bg="#00b4d8",
    activebackground="#000000",
    activeforeground="#00b4d8",
    command=info,
)
infobutton.place(x=165, y=320)


root.mainloop()
