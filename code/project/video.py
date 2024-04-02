import cv2
cap = cv2.VideoCapture('project/video/dog.mp4') #讀取影片
while True:
    ret , next = cap.read() #ret取得影片的下一張
    if ret:
        cv2.imshow('video',next)
    else:
        break
    cv2.waitKey(1)