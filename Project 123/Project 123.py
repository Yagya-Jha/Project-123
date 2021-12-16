import cv2
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

# What To write to get a secure Connection
# Setting an HTTPS context to fetch data from opnml

if(not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
  ssl._create_default_https_context=ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')["labels"]
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
n_classes = len(classes)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
X_train_scale = X_train/255
X_test_scale = X_test/255

lr = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scale, y_train)
y_prediction = lr.predict(X_test_scale)

accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)

cap = cv2.VideoCapture(0)
while (True):
  try:
    ret, frame = cap.read()
    # Drawing a box at the centre of video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape()
    upperleft = (int(width/2-56), int(height/2-56))
    bottomright = (int(width/2+56), int(height/2+56))
    cv2.rectangle(gray, upperleft, bottomright, (0,255,0), 2)
    # Region Of Interst
    roi = gray[upperleft[1]:bottomright[1], upperleft[0]:bottomright[0]]
    img_pil = Image.fromarray(roi)
    img_bw = img_pil.convert('L')
    img_rz = img_bw.resize((28, 28), Image.ANTIALIAS)
    img_inv = PIL.ImageOps.invert(img_rz)
    pixel_filter = 20
    min_pixel = np.percentile(img_inv, pixel_filter)
    img_scaled = np.clip(img_inv-min_pixel, 0, 255)
    max_pixel = np.max(img_inv)
    img_scaled = np.asarray(img_scaled)/max_pixel
    test_sample = np.array(img_scaled).reshape(1, 784)
    test_prediction = lr.predict(test_sample)
    print(f'Predicted Class is {test_prediction}')
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass
cap.release()
cv2.destroyAllWindows()