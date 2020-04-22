import cv2


def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)
    

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        
        if id == 1:
            cv2.putText(img, "RAOOF", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif id == 2:
            cv2.putText(img, "UMMI", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        # cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
        
    return coords

def detect(img, facecascade, eyescascade, img_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0)}
    
    cords = draw_boundary(img, facecascade, 1.1, 10, color["red"], "Face")
    
    if len(cords) == 4:
        roi_img = img[cords[1]:cords[1]+cords[3], cords[0]:cords[0]+cords[2]]
        # user_id = 2
        # generate_dataset(roi_img, user_id, img_id)
        
        # cords = draw_boundary(img, eyescascade, 1.1, 14, color["blue"], "eye")
    return img

def recognize(img, clf, facecascade):
    
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    cords = draw_boundary(img, facecascade, 1.1, 10, color["white"], "Face", clf)
    
    return img

video_capture = cv2.VideoCapture(0)

face_xml = cv2.CascadeClassifier("features/haarcascade_frontalface_default.xml")
eyes_xml = cv2.CascadeClassifier("features/haarcascade_eye.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("features/classifier.yml")
img_id = 0

while True: 
    _, img = video_capture.read()
    
    # img = detect(img, face_xml, eyes_xml, img_id)
    # img_id += 1
    img = recognize(img, clf, face_xml)
    cv2.imshow("face detection", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()