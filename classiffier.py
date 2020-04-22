import numpy as np
from PIL import Image
import os, cv2



def train_classifier(data_dir):
    
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    
    for image in path:
        img = Image.open(image).convert('L')
        imgNp = np.array(img, 'uint8')
        id = os.path.split(image)[1].split(".")[1]
        
        id = int(id)
        
        faces.append(imgNp)
        ids.append(id)
        
    ids = np.array(ids)
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("features/classifier.yml")
        

train_classifier("data")