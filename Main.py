from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
from tkinter.ttk import *
import geocoder
from datetime import datetime
import google.generativeai as genai
import pandas as pd
import joblib

root = tk.Tk()
root.title('KRISHI')
root.resizable(True, True)
root.geometry('1000x850')

style = Style()
style.configure('W.TButton', font=('calibri', 10, 'bold', 'underline'), foreground='red')

pil_image = Image.open("farm2.jpg")
tk_image = ImageTk.PhotoImage(pil_image)
bg_label = tk.Label(root, image=tk_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
bg_label.lower()

np.set_printoptions(suppress=True)

model = load_model("keras_model.h5", compile=False)

class_names = open("labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

root.wm_attributes("-transparentcolor", "#ab23ff")

g = geocoder.ip('me')

month_text = datetime.now().strftime("%B")

API = "AIzaSyDZEun_VK4gVV8oroCaYmpiXJACASQQH_A"

genai.configure(
    api_key=API
)

model2 = genai.GenerativeModel('gemini-pro', safety_settings=None)
chat = model2.start_chat(history=[])

model3 = joblib.load('crop_recommendation_model.joblib')

Stype = 0

if month_text in ["June", "July", "August", "September", "October"]:
    Mtype = 5
elif month_text in ["November", "December", "January", "February", "March", "April", "May"]:
    Mtype = 4


def predict_image():
    global data, class_names, Stype, response

    filename = fd.askopenfilename(
        title='Open an image file',
        initialdir='/',
        filetypes=(('Image files', '*.jpg *.jpeg *.png *.bmp *.gif'), ('All files', '*.*'))
    )

    if filename:
        image = Image.open(filename).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index] * 100
        Sname = str(class_name[2:])

        if 'black soil' in Sname.lower():
            Stype = 1
        elif "clayey soil" in Sname.lower():
            Stype = 2
        elif "sandy soil" in Sname.lower():
            Stype = 3

        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Type: {class_name[2:]} \nConfidence: {confidence_score:.2f}% \n\nYour Address: {g.address} \n\nMonth:{month_text}")
        result_text.config(state=tk.DISABLED)

        new_data = pd.DataFrame({'Type of Soil': [Stype], 'Season': [Mtype], 'Rainfall in mm': [Ramt]})
        prediction3 = model3.predict(new_data)

        question = f"which crop gows best in {class_name[2:]} in the place of {g.address} during the month of {month_text}?list or describe crops in 30 words"
        response = chat.send_message(question)
        response_text.config(state=tk.NORMAL)
        response_text.delete(1.0, tk.END)
        response_text.insert(tk.END, f"Best crop grown: {prediction3[0]}")
        response_text.config(state=tk.DISABLED)

        response_text1.config(state=tk.NORMAL)
        response_text1.delete(1.0, tk.END)
        response_text1.insert(tk.END, f"Extra: {response.text}")
        response_text1.config(state=tk.DISABLED)

def display_text():
    global entry, Ramt
    Ramt = round(int(entry.get())/100)*100

def chatbotbut():
    global entry2
    question = entry2.get()
    response_text1.config(state=tk.NORMAL)
    response = chat.send_message(question)
    response_text1.delete(1.0, tk.END)
    response_text1.insert(tk.END, response.text)
    response_text1.config(state=tk.DISABLED)

image = tk.PhotoImage(file="logo.png")
tk.Label(root, image=image, bg="#185729").pack()

label = tk.Label(root, text="Enter Annual Amount of Rainfall in mm", compound=tk.CENTER)
label.pack(pady=10)

entry = Entry(root, width=40)
entry.focus_set()
entry.pack()

ttk.Button(root, text="Okay", width=20, command=display_text).pack(pady=20)

open_button = ttk.Button(
    root,
    text='Open an Image File',
    command=predict_image
)

open_button.pack(pady=10)

result_text = tk.Text(root, height=7, width=60, state=tk.DISABLED)
result_text.pack(pady=10)

response_text = tk.Text(root, height=2, width=60, state=tk.DISABLED)
response_text.pack(pady=10)

label = tk.Label(root, text="Ask question to Krishibot", compound=tk.CENTER, fg="#000000")
label.pack(pady=10)

entry2 = Entry(root, width=40)
entry2.focus_set()
entry2.pack()

ttk.Button(root, text="Send", width=20, command=chatbotbut).pack(pady=20)

response_text1 = tk.Text(root, height=10, width=60, state=tk.DISABLED)
response_text1.pack(pady=10)

root.mainloop()