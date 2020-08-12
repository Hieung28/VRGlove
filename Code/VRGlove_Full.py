#!/usr/bin/python3

import time
import threading
import csv
import sys
import keras
import serial
import zmq
import ujson
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np
import keras_preprocessing
from keras.models import model_from_json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from timeit import default_timer as timer

from tkinter import Tk, Text, TOP, BOTTOM, LEFT, RIGHT, END, X, Y, BOTH, W, NW, N, NE, E, CENTER, StringVar, messagebox
from tkinter.ttk import Frame, Label, Entry, Button, OptionMenu, Style
# from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC

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

fieldnames_twoHand= ['R_forearm_carpal', 'R_carpal_hand', \
'R_hand_thumb' , 'R_thumb_tp_mc' , 'R_thumb_mc_pp' , 'R_thumb_pp_dp' , \
'R_hand_index' , 'R_index_mc_pp' , 'R_index_pp_mp' , 'R_index_mp_dp' , \
'R_hand_middle', 'R_middle_mc_pp', 'R_middle_pp_mp', 'R_middle_mp_dp', \
'R_hand_ring'  , 'R_ring_mc_pp'  , 'R_ring_pp_mp'  , 'R_ring_mp_dp'  , \
'R_hand_pinky' , 'R_pinky_mc_pp' , 'R_pinky_pp_mp' , 'R_pinky_mp_dp', \
'L_forearm_carpal', 'L_carpal_hand', \
'L_hand_thumb' , 'L_thumb_tp_mc' , 'L_thumb_mc_pp' , 'L_thumb_pp_dp' , \
'L_hand_index' , 'L_index_mc_pp' , 'L_index_pp_mp' , 'L_index_mp_dp' , \
'L_hand_middle', 'L_middle_mc_pp', 'L_middle_pp_mp', 'L_middle_mp_dp', \
'L_hand_ring'  , 'L_ring_mc_pp'  , 'L_ring_pp_mp'  , 'L_ring_mp_dp'  , \
'L_hand_pinky' , 'L_pinky_mc_pp' , 'L_pinky_pp_mp' , 'L_pinky_mp_dp', \
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


jointAngle=[0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0]

timeConstant=0.5


#Create a dictionary with keys are elements in joinName
jointDict = dict(zip(jointName, jointAngle))

jointNumber = len(jointName)

# characterList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
# 'A', 'B', 'C', 'D', 'Đ', 'E', 'G', 'H', \
# 'I', 'K', 'L', 'M', 'N', 'O', 'more', 'help', \
# 'play', 'S', 'T', 'U', 'V', 'X', 'Y','F']

# characterIndex = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
# '10', '11', '12', '13', '14', '15', '16', '17', \
# '18', '19', '20', '21', '22', '23', '24', '25', \
# '26', '27', '28', '29', '30', '31', '32', '33']

characterList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
'A', 'B', 'C', 'D', 'Đ', 'E', 'G', 'H', \
'I', 'K', 'L', 'M', 'N', 'O', 'more', 'help', \
'play']

characterIndex = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
'10', '11', '12', '13', '14', '15', '16', '17', \
'18', '19', '20', '21', '22', '23', '24', '25', \
'26']

ImageIndex = ['Images\\0.jpg' , 'Images\\1.jpg' , 'Images\\2.jpg' , 'Images\\3.jpg' , 'Images\\4.jpg', \
'Images\\5.jpg' , 'Images\\6.jpg' , 'Images\\7.jpg' , 'Images\\8.jpg' , 'Images\\9.jpg','Images\\A.jpg', \
'Images\\B.jpg' , 'Images\\C.jpg' , 'Images\\D.jpg' , 'Images\\Đ.jpg' , 'Images\\E.jpg' , 'Images\\G.jpg', \
'Images\\H.jpg' , 'Images\\I.jpg' , 'Images\\K.jpg' , 'Images\\L.jpg' , 'Images\\M.jpg' , 'Images\\N.jpg' , 'O.jpg', \
'Images\\P.jpg' , 'Images\\Q.jpg' , ]
####################################################################################################

class ThreadQuatProcessLeftHand(threading.Thread):
    def __init__(self, serialPort, baudRate):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        self.serialPort = serialPort
        self.baudRate = baudRate
        self.num=0.0
        # self.receiveQuatEnable = False
        
        
    # def changeSerialPort(self, serialPort):
    #     self.ser.close()
    #     self.serialPort = serialPort
    #     self.seor = serial.Serial(self.serialPort, self.baudRate)

    def run(self):
       
        try:
            with serial.Serial(self.serialPort, self.baudRate) as self.ser:
                string = ''            
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
                                global jointAngleLeft

                                def calcAngle(joint, bone1, bone2, axis):
                                    nonlocal quatPacket
                                    global jointAngleLeft
                                    if bone1 in quatPacket and bone2 in quatPacket:
                                        roll, pitch, yaw = getRelativeAngle(quatPacket[bone1], quatPacket[bone2])
                                        if axis == 'roll':                                                                                
                                            jointAngleLeft[joint] = round(roll, 3)
                                        elif axis == 'pitch':                                            
                                            jointAngleLeft[joint] = round(roll, 3)
                                        elif axis == 'yaw':                                            
                                            jointAngleLeft[joint] = round(roll, 3)
                                        
                                            

                                
                                def copyAngle(destination, source):
                                    global jointAngleLeft
                                    jointAngleLeft[destination] = jointAngleLeft[source]

                                def scaleAngle(joint, offset, scale):
                                    global jointAngleLeft
                                    jointAngleLeft[joint] = round(scale*(jointAngleLeft[joint]) + offset, 3)

                                def LimitAngle(joint, min,max):
                                    global jointAngleLeft
                                    if jointAngleLeft[joint] > max:
                                        jointAngleLeft[joint]=max
                                    if jointAngleLeft[joint] < min:
                                        jointAngleLeft[joint] = min

                                calcAngle(joint_L_forearm_carpal, hand_wist, index_mc, 'yaw')

                                if jointAngleLeft[joint_L_forearm_carpal] > 0.39:
                                    jointAngleLeft[joint_L_forearm_carpal] = 0.39
                                if jointAngleLeft[joint_L_forearm_carpal] < -0.39:
                                    jointAngleLeft[joint_L_forearm_carpal] = -0.39

                                calcAngle(joint_L_carpal_hand, hand_wist, index_mc, 'roll')

                                if jointAngleLeft[joint_L_carpal_hand] > 0.63:
                                    jointAngleLeft[joint_L_carpal_hand] = 0.63
                                if jointAngleLeft[joint_L_carpal_hand] < -0.63:
                                    jointAngleLeft[joint_L_carpal_hand] = -0.63

                                calcAngle(joint_L_hand_thumb , index_mc, thumb_mc, 'pitch')                                
                                #scaleAngle(joint_L_hand_thumb, 0, 0)

                                if jointAngleLeft[joint_L_hand_thumb] > 0.87:
                                    jointAngleLeft[joint_L_hand_thumb] = 0.87
                                if jointAngleLeft[joint_L_hand_thumb] < -0.7:
                                    jointAngleLeft[joint_L_hand_thumb] = -0.7

                                calcAngle(joint_L_thumb_tp_mc, index_mc, thumb_mc, 'yaw') #yaw
                                #scaleAngle(joint_L_thumb_tp_mc, 0, 0) #3.5scale

                                if jointAngleLeft[joint_L_thumb_tp_mc] > 0.90:
                                    jointAngleLeft[joint_L_thumb_tp_mc] = 0.90
                                if jointAngleLeft[joint_L_thumb_tp_mc] < -0.79:
                                    jointAngleLeft[joint_L_thumb_tp_mc] = -0.79

                                calcAngle(joint_L_thumb_mc_pp, thumb_mc, thumb_pp, 'roll')
                                scaleAngle(joint_L_thumb_mc_pp, -1, 1.2)

                                # if jointAngleLeft[joint_L_thumb_mc_pp] > 1.75:
                                #     jointAngleLeft[joint_L_thumb_mc_pp] = 1.75
                                # if jointAngleLeft[joint_L_thumb_mc_pp] < -0.39:
                                #     jointAngleLeft[joint_L_thumb_mc_pp] = -0.39

                                calcAngle(joint_L_thumb_pp_dp, thumb_pp, thumb_dp, 'roll')
                                scaleAngle(joint_L_thumb_pp_dp, 0, 1)

                                if jointAngleLeft[joint_L_thumb_pp_dp] > 1.75:
                                    jointAngleLeft[joint_L_thumb_pp_dp] = 1.75
                                if jointAngleLeft[joint_L_thumb_pp_dp] < -0.39:
                                    jointAngleLeft[joint_L_thumb_pp_dp] = -0.39
                                

                                calcAngle(joint_L_hand_index , index_mc, index_pp, 'yaw')

                                if jointAngleLeft[joint_L_hand_index] > 0.35:
                                    jointAngleLeft[joint_L_hand_index] = 0.35
                                if jointAngleLeft[joint_L_hand_index] < -0.16:
                                    jointAngleLeft[joint_L_hand_index] = -0.16

                                calcAngle(joint_L_index_mc_pp, index_mc, index_pp, 'roll')

                                if jointAngleLeft[joint_L_index_mc_pp] > 1.75:
                                    jointAngleLeft[joint_L_index_mc_pp] = 1.75
                                if jointAngleLeft[joint_L_index_mc_pp] < -0.39:
                                    jointAngleLeft[joint_L_index_mc_pp] = -0.39

                                calcAngle(joint_L_index_pp_mp, index_pp, index_mp, 'roll')

                                if jointAngleLeft[joint_L_index_pp_mp] > 1.75:
                                    jointAngleLeft[joint_L_index_pp_mp] = 1.75
                                if jointAngleLeft[joint_L_index_pp_mp] < -0.39:
                                    jointAngleLeft[joint_L_index_pp_mp] = -0.39

                                copyAngle(joint_L_index_mp_dp, joint_R_index_pp_mp)

                                # calcAngle(joint_R_hand_middle, index_pp, index_mp, 'pitch') #
                                # calcAngle(joint_R_middle_mc_pp, index_pp, index_mp, 'yaw') #
                                calcAngle(joint_L_hand_middle , index_mc , middle_pp, 'yaw')

                                if jointAngleLeft[joint_L_hand_middle] > 0.31:
                                    jointAngleLeft[joint_L_hand_middle] = 0.31
                                if jointAngleLeft[joint_L_hand_middle] < -0.31:
                                    jointAngleLeft[joint_L_hand_middle] = -0.31

                                calcAngle(joint_L_middle_mc_pp, index_mc , middle_pp, 'roll')

                                if jointAngleLeft[joint_L_middle_mc_pp] > 1.75:
                                    jointAngleLeft[joint_L_middle_mc_pp] = 1.75
                                if jointAngleLeft[joint_L_middle_mc_pp] < -0.39:
                                    jointAngleLeft[joint_L_middle_mc_pp] = -0.39

                                calcAngle(joint_L_middle_pp_mp, middle_pp, middle_mp, 'roll')

                                if jointAngleLeft[joint_L_middle_pp_mp] > 1.75:
                                    jointAngleLeft[joint_L_middle_pp_mp] = 1.75
                                if jointAngleLeft[joint_L_middle_pp_mp] < -0.39:
                                    jointAngleLeft[joint_L_middle_pp_mp] = -0.39

                                copyAngle(joint_L_middle_mp_dp, joint_L_middle_pp_mp)

                                if jointAngleLeft[joint_L_middle_mp_dp] > 1.75:
                                    jointAngleLeft[joint_L_middle_mp_dp] = 1.75
                                if jointAngleLeft[joint_L_middle_mp_dp] < -0.39:
                                    jointAngleLeft[joint_L_middle_mp_dp] = -0.39

                                calcAngle(joint_L_hand_ring , index_mc, ring_pp, 'yaw')
                                LimitAngle(joint_L_hand_ring,-0.35,0.16 )

                                calcAngle(joint_L_ring_mc_pp, index_mc, ring_pp, 'roll')
                                LimitAngle(joint_L_ring_mc_pp, -0.39,1.75)

                                calcAngle(joint_L_ring_pp_mp, ring_pp , ring_mp, 'roll')
                                LimitAngle(joint_L_ring_pp_mp, -0.39, 1.75)

                                copyAngle(joint_L_ring_mp_dp, joint_R_ring_pp_mp)

                                calcAngle(joint_L_hand_pinky , index_mc, pinky_pp, 'yaw')
                                LimitAngle(joint_L_hand_pinky, -0.35, 0.16)

                                calcAngle(joint_L_pinky_mc_pp, index_mc, pinky_pp, 'roll')
                                LimitAngle(joint_L_pinky_mc_pp, -0.39, 1.75)

                                calcAngle(joint_L_pinky_pp_mp, pinky_pp, pinky_mp, 'roll')
                                LimitAngle(joint_L_pinky_pp_mp, -0.39, 1.75)

                                copyAngle(joint_L_pinky_mp_dp, joint_L_pinky_pp_mp)

                                for angle in jointAngleLeft:
                                    if angle == 0.0:
                                        self.num=self.num+1
                                if self.num >=3:
                                    print(jointAngleLeft)
                                    print("0")
                                    messagebox.showerror("Error")
                                    
                                    self.shutdown_flag.set()
                                self.num = 0
                                        

                                print(jointAngleLeft)
                                print("Left")
                                

                            updateAngle()
        except:
            print("Unexpected error:", sys.exc_info()[0])
                                

####################################################################################################
class ThreadQuatProcessRightHand(threading.Thread):
        
    def __init__(self, serialPort, baudRate):
            threading.Thread.__init__(self)
            self.shutdown_flag = threading.Event()
            self.serialPort = serialPort
            self.baudRate = baudRate
            self.num=0
        
    # def changeSerialPort(self, serialPort):
    #     self.ser.close()
    #     self.serialPort = serialPort
    #     self.seor = serial.Serial(self.serialPort, self.baudRate)

    def run(self):
        #global jointAngleRight
        # while(1):
        #     jointAngle=jointAngle    
        try:
            with serial.Serial(self.serialPort, self.baudRate) as self.ser:
                string = ''            
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
                                global jointAngleRight

                                def calcAngle(joint, bone1, bone2, axis):
                                    nonlocal quatPacket
                                    global jointAngleRight
                                    if bone1 in quatPacket and bone2 in quatPacket:
                                        roll, pitch, yaw = getRelativeAngle(quatPacket[bone1], quatPacket[bone2])
                                        if axis == 'roll':                                                                                
                                            jointAngleRight[joint] = round(roll, 3)
                                        elif axis == 'pitch':                                            
                                            jointAngleRight[joint] = round(roll, 3)
                                        elif axis == 'yaw':                                            
                                            jointAngleRight[joint] = round(roll, 3)
                                
                                def copyAngle(destination, source):
                                    global jointAngleRight
                                    jointAngleRight[destination] = jointAngleRight[source]

                                def scaleAngle(joint, offset, scale):
                                    global jointAngleRight
                                    jointAngleRight[joint] = round(scale*(jointAngleRight[joint]) + offset, 3)

                                calcAngle(joint_R_forearm_carpal, hand_wist, index_mc, 'yaw')
                                calcAngle(joint_R_carpal_hand, hand_wist, index_mc, 'roll')

                                calcAngle(joint_R_hand_thumb , thumb_mc, index_mc, 'pitch')
                                scaleAngle(joint_R_hand_thumb, 0, 1.1)
                                calcAngle(joint_R_thumb_tp_mc, thumb_mc, index_mc, 'yaw')
                                scaleAngle(joint_R_thumb_tp_mc, 0, 1)
                                calcAngle(joint_R_thumb_mc_pp, thumb_mc, thumb_pp, 'roll')
                                scaleAngle(joint_R_thumb_mc_pp, 1.5, 0.7) #
                                

                                calcAngle(joint_R_thumb_pp_dp, thumb_pp, thumb_dp, 'roll')
                                scaleAngle(joint_R_thumb_pp_dp, 0, 2)
                              

                                calcAngle(joint_R_hand_index , index_mc, index_pp, 'yaw')
                                scaleAngle(joint_R_hand_index,0,0.6)
                                calcAngle(joint_R_index_mc_pp, index_mc, index_pp, 'roll')
                                calcAngle(joint_R_index_pp_mp, index_pp, index_mp, 'roll')
                                copyAngle(joint_R_index_mp_dp, joint_R_index_pp_mp)

                                calcAngle(joint_R_hand_middle , index_mc , middle_pp, 'yaw')
                                scaleAngle(joint_R_hand_middle,0,0.6)
                                calcAngle(joint_R_middle_mc_pp, index_mc , middle_pp, 'roll')
                                calcAngle(joint_R_middle_pp_mp, middle_pp, middle_mp, 'roll')
                                copyAngle(joint_R_middle_mp_dp, joint_R_middle_pp_mp)

                                calcAngle(joint_R_hand_ring , index_mc, ring_pp, 'yaw')
                                scaleAngle(joint_R_hand_ring,0,0.6)
                                calcAngle(joint_R_ring_mc_pp, index_mc, ring_pp, 'roll')
                                calcAngle(joint_R_ring_pp_mp, ring_pp , ring_mp, 'roll')
                                copyAngle(joint_R_ring_mp_dp, joint_R_ring_pp_mp)

                                calcAngle(joint_R_hand_pinky , index_mc, pinky_pp, 'yaw')
                                scaleAngle(joint_R_hand_pinky,0,0.6)
                                calcAngle(joint_R_pinky_mc_pp, index_mc, pinky_pp, 'roll')
                                calcAngle(joint_R_pinky_pp_mp, pinky_pp, pinky_mp, 'roll')
                                copyAngle(joint_R_pinky_mp_dp, joint_R_pinky_pp_mp)

                                for angle in jointAngleRight:
                                    if angle == 0.0:
                                        self.num=self.num+1
                                if self.num >=3:
                                    print(jointAngleRight)
                                    print("0")

                                    messagebox.showerror("Error")
                                   
                                    self.shutdown_flag.set()
                                   
                                self.num = 0
                                        

                                print(jointAngleRight)
                                print("right")
                                

                            updateAngle()
        except:
            print("Unexpected error:", sys.exc_info()[0])
####################################################################################################                            

####################################################################################################
class ThreadCollectData(threading.Thread):
    def __init__(self,csvFilename):
        threading.Thread.__init__(self)
        self.shutdown_flag=threading.Event()
        self.csvFilename = csvFilename
        self.trainingCharacter = '2'
        self.collectSampleEnable = False
        self.sampleCount = 0
        self.count=0
    def run(self):                
        with open(self.csvFilename, mode='a') as trainerFile:
            global fieldnames_twoHand
            global fieldnames
            global jointAngle
            global jointAngleRight
            global jointAngleLeft
            writer = csv.DictWriter(trainerFile, fieldnames=fieldnames_twoHand)
            while not self.shutdown_flag.is_set():               
                if(self.collectSampleEnable==True):
                    jointAngle=jointAngleRight+jointAngleLeft
                    for angle in jointAngle:
                        if angle == 0.0:
                            self.count =self.count+1
                    if self.count >=4:
                        print(jointAngle)
                        print("2 hand")
                        messagebox.showerror("Error")
                    else:
                       print(jointAngle)
                       newRow = dict(zip(fieldnames_twoHand, jointAngle))
                       newRow.update({'character':self.trainingCharacter})
                       writer.writerow(newRow)
                       self.sampleCount += 1
                       print('Sample count: ' + str(self.sampleCount))
                       time.sleep(0.05)
                    self.count=0

                    
                   
                    
                   

                

####################################################################################################

# class ThreadZMQPush(threading.Thread):
#     def __init__(self, IPAddress):
#         threading.Thread.__init__(self)
#         self.shutdown_flag = threading.Event()
#         self.address = IPAddress

#     def run(self):
#         context = zmq.Context()
#         # zmq_socket = context.socket(zmq.PUSH)
#         # zmq_socket.bind(self.address)
#         while not self.shutdown_flag.is_set():            
#             global jointDict
        
#             global jointAngle
#             global jointAngle_2
#             global jointAngle_1
#             if jointAngle==jointAngle_1:
#                 jointAngle=jointAngle_2
#             else:
#                 jointAngle=jointAngle_1
#             jointDict = dict(zip(jointName, jointAngle))
            # jointDictJSON = ujson.dumps(jointDict)
            # # print(jointDictJSON)
            # //zmq_socket.send_json(jointDictJSON)
            # time.sleep(0.02)
####################################################################################################
            
def ROSUpdate():
    
    rospy.init_node('joint_state_publisher', anonymous = True)
    pub = rospy.Publisher('joint_states', JointState, queue_size = 10)
    #rospy.init_node('display_robot_state', anonymous = True)
    rate = rospy.Rate(100)
    global jointAngleRight
    while not rospy.is_shutdown():
        hello_str = JointState()
        hello_str.header = Header()
        hello_str.header.stamp = rospy.Time.now()
        hello_str.name = list(jointDict.keys())
        hello_str.position = jointAngleRight
        #print(hello_str.position)
        hello_str.velocity = []
        hello_str.effort = []
        pub.publish(hello_str)
        # rospy.loginfo(hello_str)
        rate.sleep()

####################################################################################################

# class ThreadPredict(threading.Thread):
#     def __init__(self, modelFile):
#         threading.Thread.__init__(self)
#         self.shutdown_flag = threading.Event()
#         self.loadedModel = joblib.load(modelFile)
#         self.predictResult = 'N/A'
#         self.predictEnable = True        
#         self.MultiplePredictEnable = False
#         self.timeConstant=timeConstant
#         self.Count=0

#     def run(self):
#         while not self.shutdown_flag.is_set() and self.MultiplePredictEnable:
#             if self.predictEnable:
#                 global jointAngleRight
#                 global characterList                
#                 temp = np.array(jointAngleRight).reshape(1, -1)
#                 yPredict = self.loadedModel.predict(temp)
#                 self.predictResult = characterList[int(yPredict)]        
#                 app.predictResult.insert(END, self.predictResult)        
#                 print('Predict result: ' + str(self.predictResult))
#                 time.sleep(self.timeConstant)

####################################################################################################
class ThreadPredict(threading.Thread):
    def __init__(self,modelFile):
        threading.Thread.__init__(self)
        self.shutdown_flag=threading.Event()
        self.modelFile=modelFile
        self.json_file=open('twohandModel.json', 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.load_model=model_from_json(self.loaded_model_json)
        self.load_model.load_weights(modelFile)
        self.load_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
        self.predictResult = 'N/A'
        self.predictEnable = True        
        self.MultiplePredictEnable = False
        self.timeConstant=timeConstant
        self.Count=0
    def run(self):
        while not self.shutdown_flag.is_set() and self.MultiplePredictEnable:
            if self.predictEnable:
                global jointAngleRight
                global jointAngleLeft
                global jointAngle
                global characterList
                jointAngle=jointAngleRight+jointAngleLeft             
                self.Count=self.Count+1
                temp = np.array(jointAngle).reshape(1, -1)
                y_array=self.load_model.predict(temp)
                index=np.argmax(y_array)
                self.predictResult=characterList[int(index)]
                print(self.predictResult)
                app.predictResult.insert(END, self.predictResult+" ")
                time.sleep(self.timeConstant)        



####################################################################################################
class ThreadTrainingModel(threading.Thread):
    def __init__(self, modelFile, trainingDataFile):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        self.modelFile = modelFile
        self.trainingDataFile = trainingDataFile

    def train(self):
        if not self.shutdown_flag.is_set():
            myData = np.genfromtxt(self.trainingDataFile, delimiter=',')
            # Splitting data for training
            X = trainingData[:, 0:trainingData.shape[1]-1]
            # Splitting labels for training
            Y = trainingData[:, -1]

            # Splitting data in to a training set and a test set to see the accuracy
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                                test_size=0.33,
                                                                random_state=0)
            # Initiate classifier object, using One Vs Rest method
            classif = OneVsRestClassifier(SVC(C=1,
                                            kernel='poly', degree=5,
                                            coef0=0.5,
                                            tol=1e-4,
                                            shrinking=True,
                                            gamma='scale'))
            #splitting data for training
            X = myData[:, 0:myData.shape[1]-1]

            #splitting labels for training
            y = myData[:, -1]

            #Splitting data in to a training set and a test set to see the accuracy
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=0)

            num_classes=27

            #  convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            print(y_train)
            model = Sequential()
            model.add(Dense(256,input_shape=(44,), activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))

            #los,metric
            model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.SGD(lr=0.01),
                                          metrics=['accuracy'])

            history=model.fit(X_train, y_train, batch_size=512, epochs = 40)    
            score = model.evaluate(X_test, y_test, verbose=0)
            print('Test loss: %.4f'% score[0])
            print('Test accuracy %.4f'% score[1])   

            X_valid=X[28001,0:X.shape[1]].reshape(1,-1)
            print(X_valid)
            Y_valid=model.predict(X_valid)
            print(Y_valid)
            lang_index = numpy.argmax(Y_valid)
            print(lang_index)
            # load json and create model
            model_json = model.to_json()
            with open("twohandModel.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(self.modelFile)
            print("Saved model to disk")



####################################################################################################

class GUI(Frame):
    def __init__(self, jointNumber):
        super().__init__()
        self.jointNumber = jointNumber
        self.initGUI()

    def quit(self):
        self.master.quit()
        self.master.destroy()

    def initGUI(self):
        
        self.master.title('VRGlove')
        self.master.geometry('1280x720')
        self.pack(fill=BOTH, expand=True)

        # Color list: https://xkcd.com/color/rgb/
        self.colorList = ['xkcd:purple', 'xkcd:green', \
        'xkcd:blue', 'xkcd:pink', 'xkcd:brown', 'xkcd:red', \
        'xkcd:light blue', 'xkcd:teal', 'xkcd:orange', 'xkcd:light green', \
        'xkcd:magenta', 'xkcd:yellow', 'xkcd:cyan', 'xkcd:grey', \
        'xkcd:lime green', 'xkcd:light purple', 'xkcd:violet', 'xkcd:dark green', \
        'xkcd:turquoise', 'xkcd:lavender', 'xkcd:dark blue', 'xkcd:tan']

        ##################################################

        # frameCalibrate = Frame(self)
        # frameCalibrate.pack(side=RIGHT, fill=Y)

        # labelPredict = Label(frameCalibrate, text='CALIBRATE', width=32)
        # labelPredict.pack(side=TOP, padx=5, pady=5)
        s = Style()
        s.configure('My.TFrame', background='green')
        ##################################################

        framePredict = Frame(self,style='My.TFrame')
        framePredict.pack(side=RIGHT, fill=Y)

        labelPredict = Label(framePredict, text='SETTING')
        labelPredict.pack(side=TOP, padx=5, pady=5)

        # self.textPredictResult = Label(framePredict, text='N/A', font=("Arial", 50), foreground='red')
        # self.textPredictResult.pack(side=TOP, padx=5, pady=50)

        labelTrainer = Label(framePredict, text='TRAINER',style='My.TFrame')
        labelTrainer.pack(side=TOP, padx=5, pady=100)

        labelTrainingCharacter = Label(framePredict, text='Training character', )
        labelTrainingCharacter.pack(side=TOP, padx=5, pady=5)

        entryTrainingCharacter = Entry(framePredict, width=5, justify=CENTER)
        entryTrainingCharacter.pack(side=TOP, padx=5, pady=5)
        entryTrainingCharacter.insert(END, '2')

        labelSampleCount = Label(framePredict, text='Sample count')
        labelSampleCount.pack(side=TOP, padx=5, pady=5)

        self.textSampleCount = Label(framePredict, text=0)
        self.textSampleCount.pack(side=TOP, padx=5, pady=5)

        def sample():
            global threadCollectData
            global threadPredict
            if not threadCollectData.collectSampleEnable:
                threadPredict.predictEnable=False
                threadCollectData.sampleCount = 0
                threadCollectData.collectSampleEnable = True
                threadCollectData.trainingCharacter = entryTrainingCharacter.get()
                print('\nStarted sampling!')
                print('Sampling character: ' + threadCollectData.trainingCharacter + '\n')
                buttonSample.configure(text='Stop sampling')
            else:
                threadPredict.predictEnable=True
                threadCollectData.collectSampleEnable = False
                print('\nStopped sampling!\n')
                buttonSample.configure(text='Start sampling')

        buttonSample = Button(framePredict, text='Start sampling', command=sample, width=15)
        buttonSample.pack(side=TOP, padx=5, pady=5)

        def trainModel():
            global threadPredict
            threadPredict.predictEnable = False
            global threadCollectData
            if threadQuatProcess.collectSampleEnable:
                sample()
            global threadTrainingModel
            threadTrainingModel.train()
            statusText = 'Training done!\nTime elapsed: ' + str(threadTrainingModel.timeElapsed) + ' ms\n'
            statusText += 'Testing done!\nTraining accuracy: ' + str(threadTrainingModel.trainScore) + '%\n'
            statusText += 'Saving done!\nFilename: ' + threadTrainingModel.modelFile
            labelTrainingStatus.configure(text=statusText)
            threadPredict.predictEnable = True

        buttonTrain = Button(framePredict, text='Train model', command=trainModel, width=15)
        buttonTrain.pack(side=TOP, padx=5, pady=5)

        labelTrainingStatus = Label(framePredict, text='Ready')
        labelTrainingStatus.pack(side=TOP, padx=5, pady=5)

        def PredictCmd():
            global threadPredict                
            if threadPredict.predictEnable and not threadPredict.MultiplePredictEnable:
                start=timer()
                jointAngle=jointAngleRight+jointAngleLeft 
                threadPredict.Count=threadPredict.Count+1
                temp = np.array(jointAngle).reshape(1, -1)
                yPredict = threadPredict.loadedModel.predict(temp)
                img=mpimg.imread(ImageIndex[int(yPredict)])
                imgplot=plt.imshow(img)
                plt.show()
                end=timer()
                predictTime=end-start
                predictTime=round(predictTime,3)
          


                #self.textPredictResult.configure(text=predictTime)
            else:
                messagebox.showwarning("Infor", "Another process is running")
            
                
       

        def MultiplePredictCmd():
            global threadPredict
            if threadPredict.predictEnable:
                if not threadPredict.MultiplePredictEnable:
                    threadPredict.MultiplePredictEnable = True
                    buttonMultiplePredict.configure(text='Stop')
                    threadPredict.start()                                                
                else:
                    threadPredict.MultiplePredictEnable = False
                    buttonMultiplePredict.configure(text='Multiple Predict')
                    #threadPredict = ThreadPredict('finalizedModel.sav')
                    threadPredict = ThreadPredict('twoHandModel.h5')
        
       
        

        

        ##################################################

        frameGraph = Frame(self,style='My.TFrame')
        frameGraph.pack(side=RIGHT, fill=BOTH, expand=True)

        labelGraph = Label(frameGraph, text='JOINT ANGLES')
        labelGraph.pack(side=TOP, padx=5, pady=5)

        buttonPredict = Button(frameGraph, text='Single Predict', command=PredictCmd, width=15)
        buttonPredict.pack(side=BOTTOM, padx=5, pady=5)

        buttonMultiplePredict= Button(frameGraph, text='Multiple Predict', command=MultiplePredictCmd, width=15)
        buttonMultiplePredict.pack(side=BOTTOM, padx=5, pady=5)
        

        self.fig = plt.Figure(dpi = 100)
        ax = self.fig.add_subplot(111)
        ax.set(xlim = (0, 100), ylim = (-100, 100))
        canvas = FigureCanvasTkAgg(self.fig, frameGraph)
        canvas.get_tk_widget().pack(fill=BOTH, padx=5, expand=True)

        self.x = np.arange(0, 101, 1)
        self.y = []
        self.line = []
        for i in range(jointNumber):
            self.y.append([])
            for j in range(101):
                self.y[i].append(0)
            self.line.append(ax.plot(self.x, self.y[i], color=self.colorList[i], lw=1)[0])
        #########################################################
        frameMultiplePredict = Frame(self,style='My.TFrame')
        frameMultiplePredict.pack(side=RIGHT, fill=BOTH)

        LabelMultiplePredict = Label(frameMultiplePredict, text='MULTIPLE PREDICT')
        LabelMultiplePredict.pack(side=TOP, padx=5, pady=5)

        self.predictResult = Text(frameMultiplePredict, width=50,  font=("Arial",15), foreground='black')
        self.predictResult.pack(fill=BOTH, padx=5, expand=True)

        TimeLabel= Label(frameMultiplePredict, text='Time Respone', font=("Arial", 15), foreground='red')
        TimeLabel.pack(side=TOP, padx=5, pady=5)

        def UpCmd():
            global threadPredict
            global timeConstant
            if timeConstant<10:
                timeConstant=timeConstant+0.5
        
            threadPredict.timeConstant=timeConstant
            self.Time.configure(text=timeConstant)

        buttonUp=Button(frameMultiplePredict, command=UpCmd, text ='+', width=2)                    
        buttonUp.pack(side=TOP, padx=5, pady=5)

        self.Time= Label(frameMultiplePredict, text='0.5')
        self.Time.pack(side=TOP, padx=5, pady=5)

        def DownCmd():
            global threadPredict
            global timeConstant
            if timeConstant>0.5:
                timeConstant=timeConstant-0.5
            
            threadPredict.timeConstant=timeConstant
            self.Time.configure(text=timeConstant)
        
        buttonDown=Button(frameMultiplePredict, command=DownCmd, text ='-', width=2)                    
        buttonDown.pack(side=TOP, padx=5, pady=5)


        
        

        # buttonPredict = Button(frameGraph, Text='Predict')
        # buttonPredict.pack(side=TOP ,padx=5, pady=5)

        # serialPort = StringVar(frameGraph)
        # serialPort.set('/dev/rfcomm0')
        # serialPortList = ['/dev/rfcomm0', '/dev/ttyACM0', '/dev/ttyACM1']
        # serialPortOption = OptionMenu(frameGraph, serialPort, serialPortList[0], *serialPortList)
        # serialPortOption.pack(side=LEFT, padx=10, pady=10)

        # def connect():
        #     global threadQuatProcess
        #     threadQuatProcess.receiveQuatEnable = False
        #     threadQuatProcess.changeSerialPort(serialPort.get())
        #     threadQuatProcess.receiveQuatEnable = True
        #     print('\nConnected to ' + threadQuatProcess.serialPort + '!\n')

        # buttonConnect = Button(frameGraph, text='Connect', command=connect)
        # buttonConnect.pack(side=TOP, padx=5, pady=5)

        
    def updateGraph(self, i):
        for i in range(jointNumber):
            global jointAngleRight
            self.y[i].pop(0)
            self.y[i].append(jointAngleRight[i]*180.0/3.1416)
            # self.y[i].append(randint(-90, 90))
            self.line[i].set_ydata(self.y[i])
        return self.line[i],

    def updateParameters(self):
        global threadPredict
        global threadCollectData
        #self.textPredictResult.configure(text=threadPredict.predictResult)
        self.textSampleCount.configure(text=threadCollectData.sampleCount)
        self.master.after(100, self.updateParameters)

    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self.updateGraph, interval=100)
        self.master.after(100, self.updateParameters)
        self.master.mainloop()

####################################################################################################

if __name__ == '__main__':
   
    threadQuatProcessRightHand = ThreadQuatProcessRightHand('/dev/ttyACM0', 115200)
    threadQuatProcessLeftHand = ThreadQuatProcessLeftHand('/dev/ttyACM1', 115200)
    threadCollectData = ThreadCollectData('CombinedHandData.csv')
    threadPredict = ThreadPredict('twohandModel.h5')
    threadTrainingModel = ThreadTrainingModel('twohandModel.h5', 'CombinedHandData.csv')
     #threadZMQPushIP = ThreadZMQPush('tcp://192.168.1.93:5600')
    # threadZMQPushLocal = ThreadZMQPush('tcp://127.0.0.1:5600')
    
    try:
        threadQuatProcessLeftHand.start()
        threadQuatProcessRightHand.start()
        #threadZMQPushIP.start()
        # threadZMQPushLocal.start()
        # threadPredict.start()
        threadTrainingModel.start()
        threadCollectData.start()
        app = GUI(jointNumber)
        
        app.run()
        #ROSUpdate()

    except KeyboardInterrupt:
        threadQuatProcessRightHand.shutdown_flag.set()
        threadQuatProcessLeftHand.shutdown_flag.set()
        # threadZMQPushIP.shutdown_flag.set()
        # threadZMQPushLocal.shutdown_flag.set()
        threadPredict.shutdown_flag.set()
        threadTrainingModel.shutdown_flag.set()

        threadQuatProcessRightHand.join()
        threadQuatProcessLeftHand.join()
        # threadZMQPushIP.join()
        # threadZMQPushLocal.join()
        threadPredict.join()
        threadTrainingModel.join()

        app.quit()