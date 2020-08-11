""" LEFT HAND
────────────────────────────────────────
────────────────────────────────────────
────────────────────███──████───────────
────────────────────████─█████──────────
───────────███──████───███───███────────
──────────██████████───███───███────────
──────────██──████───███─────███────────
──────────██──███───████───████─────────
──────────██──██────██─────███──────────
──────────██──██──██────██████████──────
──────────██──██──██───████████████─────
──────────██──██──██──███────────███────
──────────██──██──██──███───████─███────
──────────██──██──██──██───█████─████───
──────────██──────██──────██───██████───
──────────███─────██──────██───████████─
──────────███─────────────██───██────██─
──────────███─────────────██─███─────██─
──────────███─────────────██████───████─
──────────███──────────────█████───██───
──────────███───────────────███──███────
──────────███────────────────────███────
───────────███───────────────────███────
────────────███────────────────████─────
────────────███────────────────███──────
─────────────███─────────────███────────
──────────────██────────────████────────
──────────────██───────────███──────────
──────────────████████████████──────────
──────────────████████████████──────────
────────────────────────────────────────
────────────────────────────────────────
"""
###################################################################################################
#!/usr/bin/python3
import time
import threading
import csv
import serial

import numpy as np
from random import randint

from Utility import *
from Quaternion import *
####################################################################################################

fieldnames = ['forearm_carpal', 'carpal_hand', \
'hand_thumb' , 'thumb_tp_mc' , 'thumb_mc_pp' , 'thumb_pp_dp' , \
'hand_index' , 'index_mc_pp' , 'index_pp_mp' , 'index_mp_dp' , \
'hand_middle', 'middle_mc_pp', 'middle_pp_mp', 'middle_mp_dp', \
'hand_ring'  , 'ring_mc_pp'  , 'ring_pp_mp'  , 'ring_mp_dp'  , \
'hand_pinky' , 'pinky_mc_pp' , 'pinky_pp_mp' , 'pinky_mp_dp', \
'character']

jointName = ['joint_R_forearm_carpal', 'joint_R_carpal_hand', \
'joint_R_hand_thumb' , 'joint_R_thumb_tp_mc' , 'joint_R_thumb_mc_pp' , 'joint_R_thumb_pp_dp' , \
'joint_R_hand_index' , 'joint_R_index_mc_pp' , 'joint_R_index_pp_mp' , 'joint_R_index_mp_dp' , \
'joint_R_hand_middle', 'joint_R_middle_mc_pp', 'joint_R_middle_pp_mp', 'joint_R_middle_mp_dp', \
'joint_R_hand_ring'  , 'joint_R_ring_mc_pp'  , 'joint_R_ring_pp_mp'  , 'joint_R_ring_mp_dp'  , \
'joint_R_hand_pinky' , 'joint_R_pinky_mc_pp' , 'joint_R_pinky_pp_mp' , 'joint_R_pinky_mp_dp']

jointName = ['joint_R_forearm_carpal', 'joint_R_carpal_hand', \
'joint_R_hand_thumb' , 'joint_R_thumb_tp_mc' , 'joint_R_thumb_mc_pp' , 'joint_R_thumb_pp_dp' , \
'joint_R_hand_index' , 'joint_R_index_mc_pp' , 'joint_R_index_pp_mp' , 'joint_R_index_mp_dp' , \
'joint_R_hand_middle', 'joint_R_middle_mc_pp', 'joint_R_middle_pp_mp', 'joint_R_middle_mp_dp', \
'joint_R_hand_ring'  , 'joint_R_ring_mc_pp'  , 'joint_R_ring_pp_mp'  , 'joint_R_ring_mp_dp'  , \
'joint_R_hand_pinky' , 'joint_R_pinky_mc_pp' , 'joint_R_pinky_pp_mp' , 'joint_R_pinky_mp_dp']

jointAngle = [0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0]

# jointAngle = [-0.125, 0.117, 0.6, -0.963, 1.454, 0.33, 0.141, 0.841, 0.598, 0.598, \
# -0.05, 0.97, 0.676, 0.676, -0.087, 0.962, 0.444, 0.444, -0.25, 0.743, \
# 0.49, 0.49] #0

timeConstant=0.5

ImageIndex = ['Images\\0.jpg','Images\\1.jpg','Images\\2.jpg','Images\\3.jpg','Images\\4.jpg',\
'Images\\5.jpg','Images\\6.jpg','Images\\7.jpg','Images\\8.jpg','Images\\9.jpg','Images\\A.jpg',\
'Images\\B.jpg','Images\\C.jpg','Images\\D.jpg','Images\\Đ.jpg','Images\\E.jpg','Images\\G.jpg',\
'Images\\H.jpg','Images\\I.jpg','Images\\K.jpg','Images\\L.jpg','Images\\M.jpg','Images\\N.jpg','Images\\O.jpg',\
'Images\\P.jpg','Images\\Q.jpg','Images\\R.jpg','Images\\S.jpg','Images\\T.jpg','Images\\U.jpg','Images\\V.jpg',\
'Images\\X.jpg','Images\\Y.jpg']
#Create a dictionary with keys are elements in joinName
jointDict = dict(zip(jointName, jointAngle))

jointNumber = len(jointName)

characterList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
'A', 'B', 'C', 'D', 'Đ', 'E', 'G', 'H', \
'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
'R', 'S', 'T', 'U', 'V', 'X', 'Y']

characterIndex = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
'10', '11', '12', '13', '14', '15', '16', '17', \
'18', '19', '20', '21', '22', '23', '24', '25', \
'26', '27', '28', '29', '30', '31', '32']

####################################################################################################

class ThreadQuatProcess(threading.Thread):
    def __init__(self, serialPort, baudRate, csvFilename):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        self.serialPort = serialPort
        self.baudRate = baudRate
        # self.receiveQuatEnable = False
        self.csvFilename = csvFilename
        self.trainingCharacter = '5'
        self.collectSampleEnable = False
        self.sampleCount = 0

    # def changeSerialPort(self, serialPort):
    #	 self.ser.close()
    #	 self.serialPort = serialPort
    #	 self.ser = serial.Serial(self.serialPort, self.baudRate)

    def run(self):
        with serial.Serial(self.serialPort, self.baudRate) as self.ser:
            string = ''
            with open(self.csvFilename, mode='a') as trainerFile:
                global fieldnames
                writer = csv.DictWriter(trainerFile, fieldnames=fieldnames)
                while not self.shutdown_flag.is_set():
                    byte = self.ser.read()
                    if byte != '$'.encode('ascii'):
                        string += byte.decode('ascii', 'ignore')
                        if byte == '\n'.encode('ascii'):
                            print(string)
                            string = ''
                    else:
                        quatPacket = read_packet(self.ser)
                        if quatPacket is None:
                            print('Failed to receive string start with $')
                        else:
                            def updateAngle():
                                nonlocal quatPacket
                                global jointAngle

                                def calcAngle(joint, bone1, bone2, axis):
                                    nonlocal quatPacket
                                    global jointAngle
                                    if bone1 in quatPacket and bone2 in quatPacket:
                                        roll, pitch, yaw = getRelativeAngle(quatPacket[bone1], quatPacket[bone2])
                                        if axis == 'roll':
                                            jointAngle[joint] = round(roll, 3)
                                        elif axis == 'pitch':
                                            jointAngle[joint] = round(pitch, 3)
                                        elif axis == 'yaw':
                                            jointAngle[joint] = round(yaw, 3)

                                def copyAngle(destination, source):
                                    global jointAngle
                                    jointAngle[destination] = jointAngle[source]

                                def scaleAngle(joint, offset, scale):
                                    global jointAngle
                                    jointAngle[joint] = round(scale*(jointAngle[joint]) + offset*3.1416/180, 3)

                                #angle of hand wrist
                                calcAngle(joint_R_forearm_carpal, hand_wist, index_mc, 'yaw') 	#0  #2 angle of hand_wrist
                                calcAngle(joint_R_carpal_hand, hand_wist, index_mc, 'roll')	#1

                                #angle of thumb to backhand
                                calcAngle(joint_R_hand_thumb , thumb_mc, index_mc, 'pitch')	#2  
                                #scaleAngle(joint_R_hand_thumb, -3.5, -3.5)	                #limit angle
                                calcAngle(joint_R_thumb_tp_mc, thumb_mc, index_mc, 'yaw')	#3
                                #scaleAngle(joint_R_thumb_tp_mc, 0, 3.5)                         #limit angle

                                #angle of thumb
                                calcAngle(joint_R_thumb_mc_pp, thumb_mc, thumb_pp, 'roll')	#4
                                calcAngle(joint_R_thumb_pp_dp, thumb_pp, thumb_dp, 'roll')	#5
                                #scaleAngle(joint_R_thumb_pp_dp, 0, 1.5)                         #limit angle
                                #scaleAngle(joint_R_thumb_mc_pp, 0, 1.1)                         #limit angle

                                #angle of index, middle, ring, pinky finger
                                calcAngle(joint_R_hand_index , index_mc, index_pp, 'yaw')	#6
                                calcAngle(joint_R_index_mc_pp, index_mc, index_pp, 'roll')	#7
                                calcAngle(joint_R_index_pp_mp, index_pp, index_mp, 'roll')	#8
                                copyAngle(joint_R_index_mp_dp, joint_R_index_pp_mp)		#9

                                calcAngle(joint_R_hand_middle , index_mc , middle_pp, 'yaw')	#10
                                calcAngle(joint_R_middle_mc_pp, index_mc , middle_pp, 'roll')	#11
                                calcAngle(joint_R_middle_pp_mp, middle_pp, middle_mp, 'roll')	#12
                                copyAngle(joint_R_middle_mp_dp, joint_R_middle_pp_mp)		#13

                                calcAngle(joint_R_hand_ring , index_mc, ring_pp, 'yaw')		#14
                                calcAngle(joint_R_ring_mc_pp, index_mc, ring_pp, 'roll')	#15
                                calcAngle(joint_R_ring_pp_mp, ring_pp , ring_mp, 'roll')	#16
                                copyAngle(joint_R_ring_mp_dp, joint_R_ring_pp_mp)		#17

                                calcAngle(joint_R_hand_pinky , index_mc, pinky_pp, 'yaw')	#18
                                calcAngle(joint_R_pinky_mc_pp, index_mc, pinky_pp, 'roll')	#19
                                calcAngle(joint_R_pinky_pp_mp, pinky_pp, pinky_mp, 'roll')	#20
                                copyAngle(joint_R_pinky_mp_dp, joint_R_pinky_pp_mp)		#21

                            updateAngle()
                            #print(quatPacket)
                            print(jointAngle)
####################################################################################################

if __name__ == '__main__':
    #threadQuatProcess = ThreadQuatProcess('COM4', 115200, 'trainingData.csv')
    threadQuatProcess = ThreadQuatProcess('/dev/ttyACM1', 115200, 'trainingData.csv')
    try:
        threadQuatProcess.start()
    except KeyboardInterrupt:
        threadQuatProcess.shutdown_flag.set()
        threadQuatProcess.join()
