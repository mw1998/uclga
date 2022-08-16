import cv2
import numpy as np
import os


source_path = r'TestImage/'
save_path = r'TestAreaRResults/'
#子宫部分二值化
def uterusbinary(image):

    img = image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = (img[i][j] > 0) * 255
    return img

def tumorbinary(image):

    img = image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = (img[i][j] > 0  and img[i][j] <50) * 255
    return img

#获取内部轮廓面积
def getArea(img):
    conts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    conts = conts[0] if len(conts) == 2 else conts[1]
    c = max(conts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    return area, c
    # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area < 200000:
    #         # cv2.drawContours(imgContour1, cnt, -1, (255, 0, 0), 2)
    #         return area


#读取所有图像并遍历处理
filenames = os.listdir(source_path)
i = 0
for file in filenames:
    i += 1
    print(file)
    print('{} / {}' .format(i, len(filenames)))
    img = cv2.imread(source_path + file)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imggray = imgGray.copy()
    binary = tumorbinary(imggray)
    Ubinary = uterusbinary(imgGray)
    imgresult = img.copy()
    #找到轮廓，有待改进
    Tcontours, Thierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    Ucontours, Uhierarchy = cv2.findContours(Ubinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #接收图像最大轮廓和最大轮廓的面积
    R1, c1 = getArea(Ubinary)
    R2, c2 = getArea(binary)

    #画出轮廓
    cv2.drawContours(imgresult, c2, -2, (255, 0, 0), 1)
    cv2.drawContours(imgresult, c1, -2, (255, 0, 0), 1)
    #添加注释

    cv2.putText(imgresult, 'R1=%.1f' %R1, (150, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(imgresult, 'R2=%.1f' %R2, (150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(imgresult, 'Ratio=%.3f' % (R2/R1), (150, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    # cv2.putText(imgresult, 'Ratio=%.3f' %(max(MaxRat)), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)  
    print(R2/R1)             
    ratio = '-Ratio=%.3f' % (R2/R1)

    #依次保存文件并更名
    cv2.imwrite(save_path + file + ratio + "-4-1.png", imgresult)

print('success process')


