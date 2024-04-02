import cv2

img = cv2.imread('project/img/GodTon.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier('project/face_detect.xml')
FaceReck = faceCascade.detectMultiScale(gray,1.1,3)
print(len(FaceReck))
for(x,y,w,h) in FaceReck:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow('Img',img)
cv2.waitKey(0)