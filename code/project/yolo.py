from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = 'project_practice\\project\\video\\721558923.288186.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

pause = False

# read frames
while ret:
    if not pause:
        ret, frame = cap.read()

        if ret:

            # detect objects
            # track objects
            results = model.track(frame, persist=True)

            # plot results
            # cv2.rectangle
            # cv2.putText
            frame_ = results[0].plot()

            # visualize
            cv2.imshow('frame', frame_)
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