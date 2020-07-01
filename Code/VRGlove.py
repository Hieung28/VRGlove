#!/usr/bin/python3

import time
import threading
import serial
import zmq 
import ujson
# from builtins import print
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC

from Utility import *
from Quaternion import *

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

characterList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
'A', 'B', 'C', 'D', 'Đ', 'E', 'G', 'H', \
'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
'R', 'S', 'T', 'U', 'V', 'X', 'Y']

characterIndex = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
'10', '11', '12', '13', '14', '15', '16', '17', \
'18', '19', '20', '21', '22', '23', '24', '25', \
'26', '27', '28', '29', '30', '31', '32']

jointAngle = [0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0, \
0.0, 0.0, 0.0, 0.0]

jointDict = dict(zip(jointName, jointAngle))

def trainModel(modelFile, trainingDataFile):
	trainingData = np.genfromtxt(trainingDataFile, delimiter=',')
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
	print("Training...")
	start = time.time()
	classif.fit(X_train, Y_train)
	end = time.time()
	print("Training done. Time elapsed:", end - start, "seconds")
	print("Testing...")
	trainScore = classif.score(X_test, Y_test)
	trainScore = format(trainScore*100, '.2f')
	print('Testing done. Training accuracy:', trainScore, "%")
	# Save model to disk
	print("Saving model to disk under name:", modelFile, "...")
	joblib.dump(classif, modelFile)
	print("Saving done.\n")

class ThreadQuatProcess(threading.Thread):
	def __init__(self, serialPort, baudRate, csvFilename, label, collectTrainer):
		threading.Thread.__init__(self)
		self.shutdown_flag = threading.Event()
		self.serialPort = serialPort
		self.baudRate = baudRate
		self.csvFilename = csvFilename
		self.label = label
		self.collectTrainer = collectTrainer
		self.sampleCount = 0

	def run(self):
		with serial.Serial(self.serialPort, self.baudRate) as ser:
			string = ''
			with open(self.csvFilename, mode='a') as trainerFile:
				global fieldnames
				writer = csv.DictWriter(trainerFile, fieldnames=fieldnames)
				while not self.shutdown_flag.is_set():
					byte = ser.read()
					if byte != '$'.encode('ascii'):
						string += byte.decode('ascii', 'ignore')
						if byte == '\n'.encode('ascii'):
							print(string)
							string = ''
					else:
						quatPacket = read_packet(ser)
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

							updateAngle()
							# print(quatPacket)
							print(jointAngle)

							if self.collectTrainer == True:
								newRow = dict(zip(fieldnames, jointAngle))
								newRow.update({'character':self.label})
								writer.writerow(newRow)
								self.sampleCount = self.sampleCount + 1
								print(self.sampleCount)

class ThreadPredict(threading.Thread):
	def __init__(self, filename):
		threading.Thread.__init__(self)
		self.shutdown_flag = threading.Event()
		self.filename = filename
		self.loadedModel = joblib.load(filename)

	def run(self):
		while not self.shutdown_flag.is_set():
			global jointAngle
			global characterList
			temp = np.array(jointAngle).reshape(1, -1)
			yPredict = self.loadedModel.predict(temp)
			predictResult = characterList[int(yPredict)]
			print('Predict result: ' + predictResult)
			time.sleep(0.1)

class ThreadZMQPush(threading.Thread):
	def __init__(self, IPAddress):
		threading.Thread.__init__(self)
		self.shutdown_flag = threading.Event()
		self.address = IPAddress

	def run(self):
		context = zmq.Context()
		zmq_socket = context.socket(zmq.PUSH)
		zmq_socket.bind(self.address)
		while not self.shutdown_flag.is_set():
			try:
				global jointDict
				jointDict = dict(zip(jointName, jointAngle))
				jointDictJSON = ujson.dumps(jointDict)
				# print(jointDictJSON)
				zmq_socket.send_json(jointDictJSON)
				time.sleep(0.02)
			except:
				zmq_socket.close()

if __name__ == '__main__':
	# thread1 = ThreadQuatProcess('/dev/ttyACM0', 1382400)
	thread1 = ThreadQuatProcess('/dev/rfcomm0', 1382400, 'trainingData.csv', '0', True)
	# thread2 = ThreadPredict('finalizedModel.sav')
	# thread3 = ThreadZMQPush('tcp://192.168.1.93:5600')
	thread4 = ThreadZMQPush('tcp://127.0.0.1:5600')
	try:
		thread1.start()
		# thread2.start()
		# thread3.start()
		thread4.start()
		while True:
			trainModel('finalizedModel.sav', 'trainingData.csv')
			time.sleep(2)
	except:
		thread1.shutdown_flag.set()
		thread2.shutdown_flag.set()
		thread3.shutdown_flag.set()
		thread4.shutdown_flag.set()
		thread1.join()
		thread2.join()
		thread3.join()
		thread4.join()
