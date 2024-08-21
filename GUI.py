import tkinter as tk
from tkinter import filedialog
from tkinter import*
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2


# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

# def FacialExpressionMode(json_file,weights_file):
#   with open(json_file,"r") as file:
#     loaded_model_json= file.read()
#     model=model_from_json(loaded_model_json)

#   model.load_weights(weights_file)
#   model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])  

#   return model

def FacialExpressionModel(json_file, weights_file):
    try:
        with open(json_file, "r") as file:
            loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_file)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


top=tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1=Label(top,background='#CDCDCD',font=('arial',15,'bold'))
sign_image=Label(top)

facec=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=FacialExpressionModel("model_a1.json","model.weights.h5")

EMOTIONS_LIST=["angry","disgust","fear","happy","neutral","sad","surprise"]

# def Detect(file_path):
#   global Label_packed

#   image=cv2.imread(file_path)
#   gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#   faces= facec.detectMultiScale(gray_image,1.3,5)

#   try:
#     for(x,y,w,h) in faces:
#       fc= gray_image[y:y+h,x:x+w]
#       roi=cv2.resize(fc,(48,48))
#       pred=EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
#       print("Pridicted Emotion is"+ pred)
#       label1.configure(foreground='#011638', text=pred)

#   except:
#     label1.configure(foreground='#011638',text='unable to detect')


# def show_Detect_button(file_path):
#   Detect_b= Button(top,text='Detect Emotion',command= lambda:Detect(file_path),padx=10,pady=5)
#   Detect_b.configure(background='#364156',foreground='white',font=('arial',10,'bold'))
#   Detect_b.place(relx=0.79,rely=0.46)


# def upload_image():
#   try:
#     file_path= filedialog.askopenfile()
#     uploaded=Image.open(file_path)
#     uploaded.thumbnail(((top.winfo_width()/2.3),(top.winfo_height()/2.3)))
#     im=ImageTk.PhotoImage(uploaded) 

#     sign_image.configure(image=im)
#     sign_image.image=im
#     label1.configure(text='')
#     show_Detect_button(file_path)

#   except:
#     pass


# upload= Button(top,text="Upload Image",command=upload_image,padx=10,pady=5)
# upload.configure(background='#364156',foreground='white',font=('arial',15,'bold'))
# upload.pack(side='bottom',pady=50)

# sign_image.pack(side='bottom',expand='True')
# label1.pack(side='bottom',expand='True')

# heading=Label(top,text='Emotion Detection',pady=20,font=('arial',15,'bold'))
# heading.configure(background='#CDCDCD',foreground='#364156')
# heading.pack()

# top.mainloop()

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)

    try:
        # for (x, y, w, h) in faces:
        #     fc = gray_image[y:y+h, x:x+w]
        #     roi = cv2.resize(fc, (48, 48))
        #     roi = roi.astype('float32') / 255  # Normalize the image
        #     roi = np.expand_dims(roi, axis=0)
        #     roi = np.expand_dims(roi, axis=-1)  # Add the channel dimension
        #     pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
        #     print("Predicted Emotion is " + pred)
        #     label1.configure(foreground='#011638', text=pred)
        for(x,y,w,h) in faces:
             fc= gray_image[y:y+h,x:x+w]
             roi=cv2.resize(fc,(48,48))
             pred=EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
             print("Pridicted Emotion is"+ pred)
             label1.configure(foreground='#011638', text=pred)

    except Exception as e:
        print(f"Error in detection: {e}")
        label1.configure(foreground='#011638', text='Unable to detect')

def show_Detect_button(file_path):
    Detect_b = Button(top, text='Detect Emotion', command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background='#364156',foreground='white',font=('arial',10,'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.3), (top.winfo_height()/2.3)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error uploading image: {e}")

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 15, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Emotion Detection', pady=20, font=('arial', 15, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()