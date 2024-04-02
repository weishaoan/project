import cv2
import numpy as np

# 创建一个 VideoCapture 对象，读取视频
cap = cv2.VideoCapture("project_practice\\project\\video\\721558923.288186.mp4")
A = []
# 创建一个背景分割器
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# 创建一个空列表，用于记录标记的坐标
time = 5
coordinates = []
pause = False

while True:
    if not pause:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        
        _, thresh = cv2.threshold(fg_mask, 127, 255, 0)
        thresh = cv2.erode(thresh, None, iterations=5)
        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh = cv2.medianBlur(thresh, 5)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #contours,hierarchy(image,mode,method[,offset])
        #創建一個全黑圖像
        filtered_thresh = np.zeros_like(thresh)

        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                if float(h) / w > 0.5:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    coordinates.append((x + w // 2, y + h // 2))
                    cv2.drawContours(filtered_thresh, [contour], -1, (255), thickness=cv2.FILLED)

            if len(coordinates) > time:
                coordinates.pop(0) 

       
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            if float(h) / w > 0.5:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center_x = x + w // 2
                center_y = y + h // 2
                if len(A) >= 8:
                    A.pop(0)  # 如果A已經有100個元素，移除最舊的元素
                A.append((center_x, center_y))

    

        cv2.imshow("Filtered Foreground Mask", filtered_thresh)
        cv2.imshow("Original Video", frame)
        

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):  
            while True:
                key = cv2.waitKey(1)
                if key == ord('p') or key == ord('q'):
                    break
        if key == ord('q'):
            break
for point in A:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
cap.release()
cv2.destroyAllWindows()