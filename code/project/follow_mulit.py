import cv2
import numpy as np
 
#读取视频
cap = cv2.VideoCapture('project_practice\\project\\video\\721558923.288186.mp4')
 
#读取第一帧图片，提取其特征点向量
ret,old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
 
#角点（特征点）检测 （shi-Tomasi角点检测）
old_pts = cv2.goodFeaturesToTrack(old_gray,maxCorners=100,qualityLevel=0.3,minDistance=10)
# print(old_pts)
 
#创建一个mask
mask = np.zeros_like(old_frame)
#随机颜色
color = np.random.randint(0,255,size=(100,3))
print(color[1].tolist())
 
while True:
    ret,frame = cap.read()
    if frame is None:
        break
         
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 
    #光流估计
    next_pts,status,err = cv2.calcOpticalFlowPyrLK(old_gray,gray,old_pts,None,winSize = (15,15),maxLevel=4)
#     print(len(next_pts))
 
    #哪些特征点找到了，哪些特征点没有找到
    good_new = next_pts[status == 1]
    good_old = old_pts[status == 1]
    print(good_new)
     
    #绘制特征点的轨迹
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        x1,y1 = new
        x0,y0 = old
        mask = cv2.line(mask,(x1,y1),(x0,y0),color[i].tolist(),2)
        frame = cv2.circle(frame,(x1,y1),5,color[i].tolist(),-1)
     
    img = cv2.add(mask,frame)  #把轨迹和当前帧图片融合
     
    cv2.imshow('mask',mask)
    cv2.imshow('video',img)
    key =cv2.waitKey(100)
    if key == ord('q'):
        break
         
    #更新
    old_gray = gray.copy()
    old_pts = good_new.reshape(-1,1,2)   #要把 good_new 的维度变回old_pts一样
     
cap.release()
cv2.destroyAllWindows()