from KC_bot_lib import KCMORTOR
from KC_bot_lib import BLEUART
import bluetooth
from machine import Pin
import time
from machine import Pin

buff = 0
kcbot = KCMORTOR()
pin26 = Pin(26, Pin.OUT)
pin5 = Pin(5, Pin.OUT)


def control():
    global buff
    buff = ble_uart.read().decode().strip()
    if len(buff) > 0:
        buff1 = buff[0]
        if buff1 == "T":
            kcbot.motor(1, 800)
            kcbot.motor(2, 800)
            pin5.value(0)
            pin26.value(1)
        if buff1 == "R":
            kcbot.motor(1, 800)
            kcbot.motor(2, -800)
            pin5.value(0)
            pin26.value(1)
        if buff1 == "L":
            kcbot.motor(1, -800)
            kcbot.motor(2, 800)
            pin5.value(0)
            pin26.value(1)
        if buff1 == "V":
            pin26.value(0)
            pin5.value(0)
        if buff1 == "Z":
            pin5.value(1)
            pin26.value(1)
        if buff1 == "Y":
            kcbot.motor(1, -500)
            kcbot.motor(2, -500)
            pin5.value(0)
            pin26.value(1)
        if buff1 == "N":
            kcbot.motorstop()


ble = bluetooth.BLE()
ble_uart = BLEUART(ble, name="KC_bot_1")
ble_uart.irq(handler=control)
