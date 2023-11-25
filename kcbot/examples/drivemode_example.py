from KC_bot_lib import KCMORTOR
from KC_bot_lib import BLEUART
import bluetooth
from machine import Pin
import time
from machine import Pin

buff = 0
kcbot = KCMORTOR()

# Thay đổi thông số động cơ ở đây, tsdc <= 748
tsdc = 748

command = {"M": (tsdc,tsdc),
           "L": (tsdc+25, tsdc-25),
           "K": (tsdc+50, tsdc-50),
           "I": (tsdc+75, tsdc-75),
           "H": (tsdc+100, tsdc-100),
           "G": (tsdc+125, tsdc-125),
           "F": (tsdc+150, tsdc-150),
           "E": (tsdc+175, tsdc-175),
           "D": (tsdc+200, tsdc-200),
           "C": (tsdc+225, tsdc-225),
           "B": (tsdc+250, tsdc-250),
           "A": (tsdc+275, tsdc-275),
           "N": (tsdc,tsdc),
           "O": (tsdc-25, tsdc+25),
           "P": (tsdc-50, tsdc+50),
           "Q": (tsdc-75, tsdc+75),
           "V": (tsdc-100, tsdc+100),
           "S": (tsdc-125, tsdc+125),
           "U": (tsdc-150, tsdc+150),
           "R": (tsdc-175, tsdc+175),
           "X": (tsdc-200, tsdc+200),
           "Y": (tsdc-225, tsdc+225),
           "J": (tsdc-250, tsdc+250),
           "Z": (tsdc-275, tsdc+275),
           "1": (0,0),
           "2": (-tsdc,-tsdc)}

def control():
    global buff
    buff = ble_uart.read().decode().strip()
    if len(buff) > 0:
        buff1 = buff[0]
        for i in command.keys:
            if buff1 == i:
                kcbot.motor(1,command[i][0])
                kcbot.motor(2,command[i][1]);


ble = bluetooth.BLE()
ble_uart = BLEUART(ble, name="KC_bot_1")
ble_uart.irq(handler=control)
