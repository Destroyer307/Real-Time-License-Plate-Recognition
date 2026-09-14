from ultralytics import YOLO
import cv2
import time
import numpy as np
import easyocr
import re
import cv2
from collections import deque , Counter


model = YOLO("Plate_Recognition_System/weights/best.pt")

reader = easyocr.Reader(["en"] , gpu=True)


def clean_text(text):
    text = text.upper()

    text = re.sub(r'[^0-9A-Z]' , '' , text)

    return text

setx = set()

ocr_history = deque(maxlen=5)

def majority_vote(texts):

    lengths = [len(text) for text in texts]

    common_length = Counter(lengths).most_common(1)[0][0]

    same_length = []

    for text in texts:
        if len(text) == common_length:
            same_length.append(text)

    if len(same_length) < 3:
        return None

    result = ""

    for i in range(common_length):

        characters = [text[i] for text in same_length]

        most_common_char = Counter(characters).most_common(1)[0][0]

        result += most_common_char


    return result



ptime = 0
ctime= 0

log_file = open("Plate_SAVİNG.txt" , mode="a" , encoding="utf-8")

cap = cv2.VideoCapture(r"Video dosyasının yolunu girin")

if not cap.isOpened():
    print("Kamaera açılamadı ne biçim kameran var amk")


while True:
    ret , frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame , (960,720))

    results = model(frame , conf=0.5)[0]


    for box in results.boxes[0]:

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        x1,y1,x2,y2 = map(int , box.xyxy[0])

        plate_text = frame[y1:y2,x1:x2]

        plate_text = cv2.resize(plate_text ,
                                None,
                                fx=4,
                                fy=4,
                                interpolation=cv2.INTER_LANCZOS4)

        gray = cv2.cvtColor(plate_text , cv2.COLOR_BGR2GRAY)

        cv2.imshow("PLATE CROP" , gray)

        readed_text = reader.readtext(gray)

        for _ , text , plate_conf in readed_text:

            if plate_conf < 0.7:
                continue

            asil_text = clean_text(text)

            if len(asil_text) < 5:
                continue

            if not asil_text[:2].isdigit():
                continue

            ocr_history.append(asil_text)

            if len(ocr_history) < 5:
                continue

            voted_plate = majority_vote(ocr_history)

            if voted_plate not in setx:
                setx.add(voted_plate)
                time_stamp = time.strftime("%y-%m-%d %H:%M:%S")
                log_file.write(f"{time_stamp} {voted_plate}\n")
                log_file.flush()



        cv2.rectangle(frame , (x1,y1) , (x2,y2) , (0,255,0) , 2)

    ptime = time.time()

    fps = 1 / (ptime - ctime)
    ctime = ptime

    cv2.putText(frame , f"FPS : {np.int32(fps)}" , (30,50) , cv2.FONT_HERSHEY_COMPLEX , 0.9 , (0,255,255) , 2)


    cv2.imshow("Plate Algorithm" , frame)


    if cv2.waitKey(1) % 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
