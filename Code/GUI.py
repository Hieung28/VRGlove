from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from timeit import default_timer as timer
import matplotlib.image as mpimg
import threading
import joblib

from tkinter import Tk, Text, TOP, BOTTOM, LEFT, RIGHT, END, X, Y, BOTH, W, NW, N, NE, E, CENTER, StringVar, messagebox
from tkinter.ttk import Frame, Label, Entry, Button, OptionMenu

#######################################################################

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

# jointAngle = [-0.125, 0.117, 0.6, -0.963, 1.454, 0.33, 0.141, 0.841, 0.598, 0.598, \
# -0.05, 0.97, 0.676, 0.676, -0.087, 0.962, 0.444, 0.444, -0.25, 0.743, \
# 0.49, 0.49] #O

# jointAngle= [-0.022, 0.375, -0.635, -0.738, 1.329, 0.389, -0.188, 1.115, 0.592, 0.592,\
# -0.274, 1.257, 0.739, 0.739, -0.179, 1.251, 0.848, 0.848, -0.314, 1.149,\
# 0.595, 0.595]  #0

jointAngle = [-0.039, 0.124, -0.187, -0.626, 1.205, 0.24, -0.118, \
0.159, 0.23, 0.23, -0.381, 1.186, 0.762, 0.762, \
-0.226, 1.209, 0.835, 0.835, -0.377, 1.142, 0.57, 0.57] #1

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

ImageIndex = ['Images\\0.jpg','Images\\1.jpg','Images\\2.jpg','Images\\3.jpg','Images\\4.jpg',\
'Images\\5.jpg','Images\\6.jpg','Images\\7.jpg','Images\\8.jpg','Images\\9.jpg','Images\\A.jpg',\
'Images\\B.jpg','Images\\C.jpg','Images\\D.jpg','Images\\Đ.jpg','Images\\E.jpg','Images\\G.jpg',\
'Images\\H.jpg','Images\\I.jpg','Images\\K.jpg','Images\\L.jpg','Images\\M.jpg','Images\\N.jpg','Images\\O.jpg',\
'Images\\P.jpg','Images\\Q.jpg','Images\\R.jpg','Images\\S.jpg','Images\\T.jpg','Images\\U.jpg','Images\\V.jpg',\
'Images\\X.jpg','Images\\Y.jpg']
class GUI(Frame):
	def __init__(self, jointNumber):
		super().__init__()
		self.jointNumber = jointNumber
		self.initGUI()

	def quit(self):
		self.master.quit()
		self.master.destroy()

	def initGUI(self):
		try:
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
					messagebox.showerror("Infor", "Sorry, no answer available")
					
				# global threadQuatProcess
				# if not threadQuatProcess.collectSampleEnable:
				# 	threadQuatProcess.sampleCount = 0
				# 	threadQuatProcess.collectSampleEnable = True
				# 	threadQuatProcess.trainingCharacter = entryTrainingCharacter.get()
				# 	print('\nStarted sampling!')
				# 	print('Sampling character: ' + threadQuatProcess.trainingCharacter + '\n')
				# 	buttonSample.configure(text='Stop sampling')
				# else:
				# 	threadQuatProcess.collectSampleEnable = False
				# 	print('\nStopped sampling!\n')
				# 	buttonSample.configure(text='Start sampling')

			buttonSample = Button(framePredict, text='Start sampling', command=sample)
			buttonSample.pack(side=TOP, padx=5, pady=5)

			def trainModel():
					messagebox.showerror("Infor", "Sorry, no answer available")
				# global threadPredict
				# threadPredict.predictEnable = False
				# global threadQuatProcess
				# if threadQuatProcess.collectSampleEnable:
				# 	sample()
				# global threadTrainingModel
				# threadTrainingModel.train()
				# statusText = 'Training done!\nTime elapsed: ' + str(threadTrainingModel.timeElapsed) + ' ms\n'
				# statusText += 'Testing done!\nTraining accuracy: ' + str(threadTrainingModel.trainScore) + '%\n'
				# statusText += 'Saving done!\nFilename: ' + threadTrainingModel.modelFile
				# labelTrainingStatus.configure(text=statusText)
				# threadPredict.predictEnable = True

			buttonTrain = Button(framePredict, text='Train model', command=trainModel)
			buttonTrain.pack(side=TOP, padx=5, pady=5)

			labelTrainingStatus = Label(framePredict, text='Ready')
			labelTrainingStatus.pack(side=TOP, padx=5, pady=5)

			##################################################

			frameGraph = Frame(self)
			frameGraph.pack(side=RIGHT, fill=BOTH, expand=True)

			labelGraph = Label(frameGraph, text='PREDICTION')
			labelGraph.pack(side=TOP, padx=5, pady=5)

			self.fig = plt.Figure(dpi = 100)
			ax = self.fig.add_subplot(111)
			ax.set(xlim = (0, 100), ylim = (-100, 100))
			canvas = FigureCanvasTkAgg(self.fig, frameGraph)
			canvas.get_tk_widget().pack(fill=BOTH, padx=5, expand=True)

			# self.x = np.arange(0, 101, 1)
			# self.y = []
			# self.line = []
			# for i in range(jointNumber):
			# 	self.y.append([])
			# 	for j in range(101):
			# 		self.y[i].append(0)
			# 	self.line.append(ax.plot(self.x, self.y[i], color=self.colorList[i], lw=1)[0])

			# serialPort = StringVar(frameGraph)
			# serialPort.set('/dev/rfcomm0')
			# serialPortList = ['/dev/rfcomm0', '/dev/ttyACM0', '/dev/ttyACM1']
			# serialPortOption = OptionMenu(frameGraph, serialPort, serialPortList[0], *serialPortList)
			# serialPortOption.pack(side=LEFT, padx=10, pady=10)

			def connect():
					start = timer()
					loadmodel= joblib.load('finalizedModel.sav')
					temp = np.array(jointAngle).reshape(1,-1)
					yPredict = loadmodel.predict(temp)
					end = timer()
					predictResult = characterList[int(yPredict)]
					predictTime=end-start
					predictTime=round(predictTime,3)
					self.textPredictResult.configure(text=predictTime)
					img=mpimg.imread(ImageIndex[int(yPredict)])
					imgplot=plt.imshow(img)
					plt.show()
						
			# 	global threadQuatProcess
			# 	threadQuatProcess.receiveQuatEnable = False
			# 	threadQuatProcess.changeSerialPort(serialPort.get())
			# 	threadQuatProcess.receiveQuatEnable = True
			# 	print('\nConnected to ' + threadQuatProcess.serialPort + '!\n')

			buttonConnect = Button(frameGraph, text='Connect', command=connect)
			buttonConnect.pack(side=TOP, padx=5, pady=5)

		except:
			self.quit()

	def updateGraph(self, i):
		for i in range(jointNumber):
			global jointAngle
			self.y[i].pop(0)
			self.y[i].append(jointAngle[i]*180.0/3.1416)
			# self.y[i].append(randint(-90, 90))
			self.line[i].set_ydata(self.y[i])
		return self.line[i],

	#def updateParameters(self):
		# global threadPredict
		# global threadQuatProcess
		# self.textPredictResult.configure(text=threadPredict.predictResult)
		# self.textSampleCount.configure(text=threadQuatProcess.sampleCount)
		# self.master.after(100, self.updateParameters)
		#messagebox.showerror("Infor", "Sorry, no answer available")

	def run(self):
		self.ani = animation.FuncAnimation(self.fig, self.updateGraph, interval=100)
		#self.master.after(100, self.updateParameters)
		self.master.mainloop()

if __name__ == '__main__':
	try:
		app = GUI(jointNumber)
		app.run()
	except KeyboardInterrupt:
		app.quit()
