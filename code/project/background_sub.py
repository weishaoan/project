import cv2
import numpy as np

# 创建一个 VideoCapture 对象，读取视频
cap = cv2.VideoCapture("project_practice\\project\\video\\27.mp4")

# 创建一个背景分割器
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# 创建一个空列表，用于记录标记的坐标
max_length = 2
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

        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

       
        filtered_thresh = np.zeros_like(thresh)

        
        for contour in contours:
            area = cv2.contourArea(contour)

            
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                if float(h) / w > 0.5:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    
                    coordinates.append((x + w // 2, y + h // 2))

                    cv2.drawContours(filtered_thresh, [contour], -1, (255), thickness=cv2.FILLED)

            if len(coordinates) > max_length:
                coordinates.pop(0) 

       
        for i in range(1, len(coordinates)):
            cv2.line(frame, coordinates[i - 1], coordinates[i], (0, 0, 255), 1)

        cv2.imshow("Original Video", frame)
        cv2.imshow("Filtered Foreground Mask", filtered_thresh)

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

cap.release()
cv2.destroyAllWindows()