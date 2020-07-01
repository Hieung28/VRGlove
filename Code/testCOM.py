import serial
import serial.tools.list_ports
from struct import *
import sys
import logging as log
#import datetime as dt
import time

####################################################################################################

def scanCOM():
    i=0
    ports = list(serial.tools.list_ports.comports())    #Get list of all com port
    for p in ports:                                     #Check every component in the port list
        print (p)
        i=i+1
    if i==0:
        print("Scan done: NO port found ", end ='')
        print(i)
    else:
        print("Scan done: ", end ='')
        print(i)

####################################################################################################

def getVR():
    ports = list(serial.tools.list_ports.comports())    #Get list of all com port
    for p in ports:                                     #Check every component in the port list
        print (p)
        if ("Maple" in str(p)) and ("Serial" in str(p)):#Search for specific name
            temp = str(p).split(" -")                   #Get string name
            print ("Connecting... ", end ='')
            print (temp[0])
            return temp[0]
        else:
            print ("Can't connect")   
    

####################################################################################################


####################################################################################################
    
scanCOM()
comPort = getVR()
#print(comPort)
#vr = serial.Serial(comPort,115200)

#while(1):
#    print(vr.read(1))

