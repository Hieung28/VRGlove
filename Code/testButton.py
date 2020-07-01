# importing only those functions 
# which are needed 
from tkinter import * 
from tkinter.ttk import Frame, Label, Entry, OptionMenu
  
# creating tkinter window 
root = Frame()
  
# Adding widgets to the root window 
Label(root, text = 'GeeksforGeeks', font =( 
  'Verdana', 15)).pack(side = TOP, pady = 10) 
  
# Creating a photoimage object to use image 
photo = PhotoImage(file = r"Button Images\up.png") 
  
# here, image option is used to 
# set image on button 
Button(root, text = 'Click Me !', image = photo).pack(side = TOP) 
  
mainloop() 