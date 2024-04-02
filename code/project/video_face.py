import cv2

# 載入人臉檢測分類器
faceCascade = cv2.CascadeClassifier('project_practice/project/face_detect.xml')

# 選擇影片檔案
video_path = 'project_practice/project/video/video.mp4'

# 開啟影片檔案
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # 讀取一幀(frame)
    ret, frame = cap.read()

    # 如果成功讀取影格
    if ret:
        # 將影格轉換為灰度影像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 進行人臉檢測
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 繪製人臉矩形框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 顯示影格
        cv2.imshow('Video', frame)

        # 按 'q' 鍵退出迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 釋放資源與關閉視窗
cap.release()
cv2.destroyAllWindows()
