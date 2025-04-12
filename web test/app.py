from flask import Flask,request
import os
from flask import render_template
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__,static_folder='public', static_url_path='/public')

_LINK_IMG_MAU =  "public/up_img/anh_mau.png"
_LINK_IMG_PRE =  "public/pre_img/anh_pre.png"


model = YOLO("model/best.pt") 

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        # Lưu ảnh gốc
        file.save(_LINK_IMG_MAU)
        frame = cv2.imread(_LINK_IMG_MAU)
        results = model(frame)

        detections = []
        for result in results:
            for i in range(len(result.boxes.xyxy)):
                x_min, y_min, x_max, y_max = np.array(result.boxes.xyxy)[0]

                score = round(float(result.boxes.conf[0]),2)
                # Tracking chỉ lấy các bbox có độ tin cậy cao
                if score > 0.7:
                    detections.append([x_min, y_min, x_max, y_max, score])
                    cls_ = result.boxes.cls[0].numpy()

        # Convert thành numpy array để input vào SORT
        detections = np.array(detections)
        
        if(detections.size > 0):

            # Vẽ các bounding box và ID
            for obj in detections:
                x_min, y_min, x_max, y_max, _ = obj.astype(int)
                if(result.boxes.cls.numpy()[0]==0):
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                    cv2.putText(frame, f"Role: {cls_} Score: {score}", (0,15),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                    cv2.putText(frame, f"Role: {cls_} Score: {score}", (0, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)

        cv2.imwrite(_LINK_IMG_PRE,frame)

        return render_template('index.html',
                               original_image=_LINK_IMG_MAU,
                               predicted_image=_LINK_IMG_PRE)
    
    # GET request ban đầu
    return render_template('index.html', original_image=None, predicted_image=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)