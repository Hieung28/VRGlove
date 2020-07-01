#!/usr/bin/python3

import time
import threading
import csv

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
from tkinter.ttk import Frame, Label, Entry, Button, OptionMenu
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

# jointAngle = [0.0, 0.0, \
# 0.0, 0.0, 0.0, 0.0, \
# 0.0, 0.0, 0.0, 0.0, \
# 0.0, 0.0, 0.0, 0.0, \
# 0.0, 0.0, 0.0, 0.0, \
# 0.0, 0.0, 0.0, 0.0]

# jointAngle_1 = [-0.125, 0.117, 0.6, -0.963, 1.454, 0.33, 0.141, 0.841, 0.598, 0.598, \
# -0.05, 0.97, 0.676, 0.676, -0.087, 0.962, 0.444, 0.444, -0.25, 0.743, \
# 0.49, 0.49] #0

# jointAngle_2 = [-0.039, 0.124, -0.187, -0.626, 1.205, 0.24, -0.118, \
# 0.159, 0.23, 0.23, -0.381, 1.186, 0.762, 0.762, \
# -0.226, 1.209, 0.835, 0.835, -0.377, 1.142, 0.57, 0.57] #1

jointAngle=[-0.445, -0.281, 1.085, 0.145, 2.736, 0.013, 0.581, -0.004, 0.15, 0.15, 0.805, 0.441, 1.151, 1.151, -0.03, 0.787, 1.187, 1.187, -0.771, 1.652, 0.826, 0.826]

timeConstant=0.5

# ImageIndex = ['Images\\0.jpg' , 'Images\\1.jpg' , 'Images\\2.jpg' , 'Images\\3.jpg' , 'Images\\4.jpg', \
# 'Images\\5.jpg' , 'Images\\6.jpg' , 'Images\\7.jpg' , 'Images\\8.jpg' , 'Images\\9.jpg','Images\\A.jpg', \
# 'Images\\B.jpg' , 'Images\\C.jpg' , 'Images\\D.jpg' , 'Images\\Đ.jpg' , 'Images\\E.jpg' , 'Images\\G.jpg', \
# 'Images\\H.jpg' , 'Images\\I.jpg' , 'Images\\K.jpg' , 'Images\\L.jpg' , 'Images\\M.jpg' , 'Images\\N.jpg' , 'Images\\O.jpg', \
# 'Images\\P.jpg' , 'Images\\Q.jpg' , 'Images\\R.jpg' , 'Images\\S.jpg' , 'Images\\T.jpg' , 'Images\\U.jpg' , 'Images\\V.jpg', \
# 'Images\\X.jpg' , 'Images\\Y.jpg']
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
	#	 self.seor = serial.Serial(self.serialPort, self.baudRate)

	def run(self):
		global jointAngle
		while(1):
			jointAngle=jointAngle	
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
											jointAngle[joint] = round(roll, 3)
										elif axis == 'yaw':											
											jointAngle[joint] = round(roll, 3)
								
								def copyAngle(destination, source):
									global jointAngle
									jointAngle[destination] = jointAngle[source]

								def scaleAngle(joint, offset, scale):
									global jointAngle
									jointAngle[joint] = round(scale*(jointAngle[joint]) + offset*3.1416/180, 3)

								calcAngle(joint_R_forearm_carpal, hand_wist, index_mc, 'yaw')
								calcAngle(joint_R_carpal_hand, hand_wist, index_mc, 'roll')

								calcAngle(joint_R_hand_thumb , thumb_mc, index_mc, 'pitch')
								scaleAngle(joint_R_hand_thumb, -35, -3.5)
								calcAngle(joint_R_thumb_tp_mc, thumb_mc, index_mc, 'yaw')
								scaleAngle(joint_R_thumb_tp_mc, 0, 3.5)
								calcAngle(joint_R_thumb_mc_pp, thumb_mc, thumb_pp, 'roll')
								calcAngle(joint_R_thumb_pp_dp, thumb_pp, thumb_dp, 'roll')
								scaleAngle(joint_R_thumb_pp_dp, 0, 1.5)
								scaleAngle(joint_R_thumb_mc_pp, 0, 1.1)

								calcAngle(joint_R_hand_index , index_mc, index_pp, 'yaw')
								calcAngle(joint_R_index_mc_pp, index_mc, index_pp, 'roll')
								calcAngle(joint_R_index_pp_mp, index_pp, index_mp, 'roll')
								copyAngle(joint_R_index_mp_dp, joint_R_index_pp_mp)

								# calcAngle(joint_R_hand_middle, index_pp, index_mp, 'pitch') #
								# calcAngle(joint_R_middle_mc_pp, index_pp, index_mp, 'yaw') #
								calcAngle(joint_R_hand_middle , index_mc , middle_pp, 'yaw')
								calcAngle(joint_R_middle_mc_pp, index_mc , middle_pp, 'roll')
								calcAngle(joint_R_middle_pp_mp, middle_pp, middle_mp, 'roll')
								copyAngle(joint_R_middle_mp_dp, joint_R_middle_pp_mp)

								calcAngle(joint_R_hand_ring , index_mc, ring_pp, 'yaw')
								calcAngle(joint_R_ring_mc_pp, index_mc, ring_pp, 'roll')
								calcAngle(joint_R_ring_pp_mp, ring_pp , ring_mp, 'roll')
								copyAngle(joint_R_ring_mp_dp, joint_R_ring_pp_mp)

								calcAngle(joint_R_hand_pinky , index_mc, pinky_pp, 'yaw')
								calcAngle(joint_R_pinky_mc_pp, index_mc, pinky_pp, 'roll')
								calcAngle(joint_R_pinky_pp_mp, pinky_pp, pinky_mp, 'roll')
								copyAngle(joint_R_pinky_mp_dp, joint_R_pinky_pp_mp)

								print(jointAngle)

							updateAngle()
							#ROSUpdate()
							#jointDict = dict(zip(jointName, jointAngle))
							# print(quatPacket)
							# print(jointAngle)

							if self.collectSampleEnable == True:
								newRow = dict(zip(fieldnames, jointAngle))
								newRow.update({'character':self.trainingCharacter})
								writer.writerow(newRow)
								self.sampleCount += 1
								print('Sample count: ' + str(self.sampleCount))

####################################################################################################

# class ThreadZMQPush(threading.Thread):
# 	def __init__(self, IPAddress):
# 		threading.Thread.__init__(self)
# 		self.shutdown_flag = threading.Event()
# 		self.address = IPAddress

# 	def run(self):
# 		context = zmq.Context()
# 		# zmq_socket = context.socket(zmq.PUSH)
# 		# zmq_socket.bind(self.address)
# 		while not self.shutdown_flag.is_set():			
# 			global jointDict
		
# 			global jointAngle
# 			global jointAngle_2
# 			global jointAngle_1
# 			if jointAngle==jointAngle_1:
# 				jointAngle=jointAngle_2
# 			else:
# 				jointAngle=jointAngle_1
# 			jointDict = dict(zip(jointName, jointAngle))
			# jointDictJSON = ujson.dumps(jointDict)
			# # print(jointDictJSON)
			# //zmq_socket.send_json(jointDictJSON)
			# time.sleep(0.02)
			
def ROSUpdate():
	
	rospy.init_node('joint_state_publisher', anonymous = True)
	pub = rospy.Publisher('joint_states', JointState, queue_size = 10)
	#rospy.init_node('display_robot_state', anonymous = True)
	rate = rospy.Rate(100)
	global jointAngle
	while not rospy.is_shutdown():
		hello_str = JointState()
		hello_str.header = Header()
		hello_str.header.stamp = rospy.Time.now()
		hello_str.name = list(jointDict.keys())
		hello_str.position = jointAngle
		print(hello_str.position)
		hello_str.velocity = []
		hello_str.effort = []
		pub.publish(hello_str)
		# rospy.loginfo(hello_str)
		rate.sleep()

####################################################################################################

# class ThreadPredict(threading.Thread):
# 	def __init__(self, modelFile):
# 		threading.Thread.__init__(self)
# 		self.shutdown_flag = threading.Event()
# 		self.loadedModel = joblib.load(modelFile)
# 		self.predictResult = 'N/A'
# 		self.predictEnable = True		
# 		self.MultiplePredictEnable = False
# 		self.timeConstant=timeConstant

# 	def run(self):
# 		while not self.shutdown_flag.is_set() and self.MultiplePredictEnable:
# 			if self.predictEnable:
# 				global jointAngle
# 				global characterList				
# 				temp = np.array(jointAngle).reshape(1, -1)
# 				yPredict = self.loadedModel.predict(temp)
# 				self.predictResult = characterList[int(yPredict)]		
# 				app.predictResult.insert(END, self.predictResult)		
# 				# print('Predict result: ' + str(self.predictResult))
# 				time.sleep(self.timeConstant)

####################################################################################################
class ThreadPredict(threading.Thread):
	def __init__(self,modelFile):
		threading.Thread.__init__(self)
		self.shutdown_flag=threading.Event()
		self.json_file=open('model.json', 'r')
		self.loaded_model_json = self.json_file.read()
		self.json_file.close()
		self.load_model=model_from_json(self.loaded_model_json)
		self.load_model.load_weights(modelFile)
		self.load_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1),
              metrics=['accuracy'])
		self.predictResult = 'N/A'
		self.predictEnable = True		
		self.MultiplePredictEnable = False
		self.timeConstant=timeConstant
	def run(self):
		while not self.shutdown_flag.is_set() and self.MultiplePredictEnable:
			if self.predictEnable:
				global jointAngle
				global characterList
				temp = np.array(jointAngle).reshape(1, -1)
				y_array=self.load_model.predict(temp)
				index=np.argmax(y_array)
				self.predictResult=characterList[int(index)]
				app.predictResult.insert(END, self.predictResult)
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
			trainingData = np.genfromtxt(self.trainingDataFile, delimiter=',')
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
			# Begin training and testing
			print('\nTraining...')
			start = time.time()
			classif.fit(X_train, Y_train)
			end = time.time()
			self.timeElapsed = format((end - start)*1000.0, '.2f')
			print('Training done! Time elapsed:', format((end - start)*1000.0, '.2f'), 'ms')
			print('Testing...')
			self.trainScore = classif.score(X_test, Y_test)
			self.trainScore = format(self.trainScore*100, '.2f')
			print('Testing done! Training accuracy:', self.trainScore, '%')
			# Save model to disk
			print('Saving model to disk under name:', self.modelFile, '...')
			joblib.dump(classif, self.modelFile)
			print('Saving done!\n')

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

		##################################################

		framePredict = Frame(self)
		framePredict.pack(side=RIGHT, fill=Y)

		labelPredict = Label(framePredict, text='PREDICT')
		labelPredict.pack(side=TOP, padx=5, pady=5)

		self.textPredictResult = Label(framePredict, text='N/A', font=("Arial", 50), foreground='red')
		self.textPredictResult.pack(side=TOP, padx=5, pady=50)

		labelTrainer = Label(framePredict, text='TRAINER')
		labelTrainer.pack(side=TOP, padx=5, pady=5)

		labelTrainingCharacter = Label(framePredict, text='Training character')
		labelTrainingCharacter.pack(side=TOP, padx=5, pady=5)

		entryTrainingCharacter = Entry(framePredict, width=5, justify=CENTER)
		entryTrainingCharacter.pack(side=TOP, padx=5, pady=5)
		entryTrainingCharacter.insert(END, '5')

		labelSampleCount = Label(framePredict, text='Sample count')
		labelSampleCount.pack(side=TOP, padx=5, pady=5)

		self.textSampleCount = Label(framePredict, text=0)
		self.textSampleCount.pack(side=TOP, padx=5, pady=5)

		def sample():
			global threadQuatProcess
			global threadPredict
			if not threadQuatProcess.collectSampleEnable:
				threadPredict.predictEnable=False
				threadQuatProcess.sampleCount = 0
				threadQuatProcess.collectSampleEnable = True
				threadQuatProcess.trainingCharacter = entryTrainingCharacter.get()
				print('\nStarted sampling!')
				print('Sampling character: ' + threadQuatProcess.trainingCharacter + '\n')
				buttonSample.configure(text='Stop sampling')
			else:
				threadPredict.predictEnable=True
				threadQuatProcess.collectSampleEnable = False
				print('\nStopped sampling!\n')
				buttonSample.configure(text='Start sampling')

		buttonSample = Button(framePredict, text='Start sampling', command=sample, width=15)
		buttonSample.pack(side=TOP, padx=5, pady=5)

		def trainModel():
			global threadPredict
			threadPredict.predictEnable = False
			global threadQuatProcess
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
				temp = np.array(jointAngle).reshape(1, -1)
				yPredict = threadPredict.loadedModel.predict(temp)
				img=mpimg.imread(ImageIndex[int(yPredict)])
				imgplot=plt.imshow(img)
				plt.show()
				end=timer()
				predictTime=end-start
				predictTime=round(predictTime,3)
				self.textPredictResult.configure(text=predictTime)
			else:
				messagebox.showwarning("Infor", "Another process is running")
			
				
		buttonPredict = Button(framePredict, text='Single Predict', command=PredictCmd, width=15)
		buttonPredict.pack(side=TOP, padx=5, pady=5)

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
					threadPredict = ThreadPredict('finalizedModel.sav')

		
		buttonMultiplePredict= Button(framePredict, text='Multiple Predict', command=MultiplePredictCmd, width=15)
		buttonMultiplePredict.pack(side=TOP, padx=5, pady=5)
		
		

		

		##################################################

		frameGraph = Frame(self)
		frameGraph.pack(side=RIGHT, fill=BOTH, expand=True)

		labelGraph = Label(frameGraph, text='JOINT ANGLES')
		labelGraph.pack(side=TOP, padx=5, pady=5)

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
		frameMultiplePredict = Frame(self)
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
		#	 global threadQuatProcess
		#	 threadQuatProcess.receiveQuatEnable = False
		#	 threadQuatProcess.changeSerialPort(serialPort.get())
		#	 threadQuatProcess.receiveQuatEnable = True
		#	 print('\nConnected to ' + threadQuatProcess.serialPort + '!\n')

		# buttonConnect = Button(frameGraph, text='Connect', command=connect)
		# buttonConnect.pack(side=TOP, padx=5, pady=5)

		
	def updateGraph(self, i):
		for i in range(jointNumber):
			global jointAngle
			self.y[i].pop(0)
			self.y[i].append(jointAngle[i]*180.0/3.1416)
			# self.y[i].append(randint(-90, 90))
			self.line[i].set_ydata(self.y[i])
		return self.line[i],

	def updateParameters(self):
		global threadPredict
		global threadQuatProcess
		#self.textPredictResult.configure(text=threadPredict.predictResult)
		self.textSampleCount.configure(text=threadQuatProcess.sampleCount)
		self.master.after(100, self.updateParameters)

	def run(self):
		self.ani = animation.FuncAnimation(self.fig, self.updateGraph, interval=100)
		self.master.after(100, self.updateParameters)
		self.master.mainloop()

####################################################################################################

if __name__ == '__main__':
	#threadQuatProcess = ThreadQuatProcess('/dev/rfcomm0'', 115200, 'trainingData.csv')
	#threadQuatProcess = ThreadQuatProcess('COM8', 115200, 'TrainingData.csv')
	threadQuatProcess = ThreadQuatProcess('/dev/ttyACM0', 115200, 'trainingData.csv')
	#threadZMQPushIP = ThreadZMQPush('tcp://192.168.1.93:5600')
	# threadZMQPushLocal = ThreadZMQPush('tcp://127.0.0.1:5600')
	#threadPredict = ThreadPredict('finalizedModel.sav')
	threadPredict = ThreadPredict('model.h5')
	threadTrainingModel = ThreadTrainingModel('finalizedModel.sav', 'TrainingData.csv')
	
	try:
		threadQuatProcess.start()
		#threadZMQPushIP.start()
		# threadZMQPushLocal.start()
		# threadPredict.start()
		threadTrainingModel.start()
		
		app = GUI(jointNumber)
		
		#3app.run()
		ROSUpdate()

	except KeyboardInterrupt:
		threadQuatProcess.shutdown_flag.set()
		# threadZMQPushIP.shutdown_flag.set()
		# threadZMQPushLocal.shutdown_flag.set()
		threadPredict.shutdown_flag.set()
		threadTrainingModel.shutdown_flag.set()

		threadQuatProcess.join()
		# threadZMQPushIP.join()
		# threadZMQPushLocal.join()
		threadPredict.join()
		threadTrainingModel.join()

		app.quit()