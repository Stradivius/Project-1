from KC_bot_lib import KCMORTOR
from KC_bot_lib import BLEUART
import bluetooth
from machine import Pin
import time
from machine import Pin

buff = 0
kcbot = KCMORTOR()


def control():
    global buff
    buff = ble_uart.read().decode().strip()
    if len(buff) > 0:
        buff1 = buff[0]
        if buff1 == "M":
            kcbot.motor(1, 800)
            kcbot.motor(2, 800)
        if buff1 == "L":
            kcbot.motor(1, 825)
            kcbot.motor(2, 775)
        if buff1 == "K":
            kcbot.motor(1, 850)
            kcbot.motor(2, 750)
        if buff1 == "I":
            kcbot.motor(1, 875)
            kcbot.motor(2, 725)
        if buff1 == "H":
            kcbot.motor(1, 900)
            kcbot.motor(2, 700)
        if buff1 == "G":
            kcbot.motor(1, 925)
            kcbot.motor(2, 675)
        if buff1 == "F":
            kcbot.motor(1, 950)
            kcbot.motor(2, 650)
        if buff1 == "E":
            kcbot.motor(1, 975)
            kcbot.motor(2, 625)
        if buff1 == "D":
            kcbot.motor(1, 1000)
            kcbot.motor(2, 600)
        if buff1 == "C":
            kcbot.motor(1, 1025)
            kcbot.motor(2, 575)
        if buff1 == "B":
            kcbot.motor(1, 1050)
            kcbot.motor(2, 550)
        if buff1 == "A":
            kcbot.motor(1, 1075)
            kcbot.motor(2, 525)
        if buff1 == "N":
            kcbot.motor(2, 800)
            kcbot.motor(1, 800)
        if buff1 == "O":
            kcbot.motor(2, 825)
            kcbot.motor(1, 775)
        if buff1 == "P":
            kcbot.motor(2, 850)
            kcbot.motor(1, 750)
        if buff1 == "Q":
            kcbot.motor(2, 875)
            kcbot.motor(1, 725)
        if buff1 == "V":
            kcbot.motor(2, 900)
            kcbot.motor(1, 700)
        if buff1 == "S":
            kcbot.motor(2, 925)
            kcbot.motor(1, 675)
        if buff1 == "U":
            kcbot.motor(2, 950)
            kcbot.motor(1, 650)
        if buff1 == "R":
            kcbot.motor(2, 975)
            kcbot.motor(1, 625)
        if buff1 == "X":
            kcbot.motor(2, 1000)
            kcbot.motor(1, 600)
        if buff1 == "Y":
            kcbot.motor(2, 1025)
            kcbot.motor(1, 575)
        if buff1 == "J":
            kcbot.motor(2, 1050)
            kcbot.motor(1, 550)
        if buff1 == "Z":
            kcbot.motor(2, 1075)
            kcbot.motor(1, 525)
        if buff1 == "1":
            kcbot.motorstop();
        

ble = bluetooth.BLE()
ble_uart = BLEUART(ble, name="KC_bot_1")
ble_uart.irq(handler=control)
