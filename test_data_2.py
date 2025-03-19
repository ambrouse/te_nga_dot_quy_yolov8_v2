from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort 
import os
model = YOLO("yolo11n-pose.pt") 
tracker = Sort()
link_ = "data/1260/rgb/"


def letterbox_resize(img, target_size=(640, 640)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)  # Tính tỷ lệ resize để giữ nguyên tỷ lệ gốc
    new_w, new_h = int(w * scale), int(h * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tạo ảnh nền màu đen
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Tính vị trí để dán ảnh vào giữa
    top_left_x = (target_size[0] - new_w) // 2
    top_left_y = (target_size[1] - new_h) // 2

    canvas[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = img_resized

    return canvas, scale, top_left_x, top_left_y  # Trả về scale & padding để điều chỉnh bbox


for i in os.listdir(link_):
    frame = cv2.imread(link_+i)
    frame,_,_,_ = letterbox_resize(frame, (640, 640))
    img_resize_= frame.copy()
    height,width,_ = frame.shape
    results = model(frame)
    box_ = np.array([])
    for result in results:
        detections = []
        for result in results:
            for i in range(len(result.boxes.xyxy)):
                x_min, y_min, x_max, y_max = np.array(result.boxes.xyxy)[0]
                score = result.boxes.conf[0].numpy()

                # Tracking chỉ lấy các bbox có độ tin cậy cao
                if score > 0.7:
                    detections.append([x_min, y_min, x_max, y_max, score])
                    box_ = result.boxes.xywh[0].numpy()
                    box_ = box_ / np.array([width, height, width, height])

        # Convert thành numpy array để input vào SORT
        detections = np.array(detections)
        
        if(detections.size > 0):
            tracked_objects = tracker.update(detections)

            # Vẽ các bounding box và ID
            for obj in tracked_objects:
                x_min, y_min, x_max, y_max, track_id = obj.astype(int)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Tracking", frame)
                cv2.waitKey(1)
                a = input("Nhập nhãn cho 60 frame ảnh : ")
                if(a=="0"):
                    with open(f"data_decode/label/train/label_{len(os.listdir("data_decode/label/train"))}.txt", "w", encoding="utf-8") as file:
                        a = [str(box_[i])+" " if i<len(box_)-1 else str(box_[i]) for i in range(len(box_))]
                        a.insert(0, "0 ")  
                        file.writelines(a)
                        cv2.imwrite(f"data_decode/img/train/img_{len(os.listdir("data_decode/label/train"))}.jpg", img_resize_)
                elif(a=="1"):
                    with open(f"data_decode/label/train/label_{len(os.listdir("data_decode/label/train"))}.txt", "w", encoding="utf-8") as file:
                        a = [str(box_[i])+" " if i<len(box_)-1 else str(box_[i]) for i in range(len(box_))]
                        a.insert(0, "1 ")  
                        file.writelines(a)
                        cv2.imwrite(f"data_decode/img/train/img_{len(os.listdir("data_decode/label/train"))}.jpg", img_resize_)
cv2.destroyAllWindows()
    


