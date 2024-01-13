import cv2
import numpy as np
from PIL import Image
import os

# Đường dẫn đến thư mục chứa dữ liệu
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Đang training dữ liệu")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Lưu file vào thư mục trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 

# In số lượng khuôn mặt được training và thoát
print("\n [INFO] {0} khuôn mặt được training. Thoát".format(len(np.unique(ids))))