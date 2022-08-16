import cv2
import numpy as np
import math
import os
from skimage.draw import line


source_path = r'TestImage/'
save_path = r'TestFitEllipseResults/'




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


# Find largest contour
def fLarContour(img):
    contour = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[0] if len(contour) == 2 else contour[1]
    big_contour = max(contour, key=cv2.contourArea)
    return big_contour

# Fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree
# def fitEllipse(bigcnt):
#     ellipset = cv2.fitEllipse(bigcnt)
#     (x, y), (d1, d2), angle = ellipset
#     return (x, y), (d1, d2), angle

# Draw ellipse
def drawEllipse(image, ellipset, ellipseu):
    # cv2.ellipse(image, ellipset, (255, 64, 128), 2)
    cv2.ellipse(image, ellipseu, (128, 192, 255), 2)

# Draw circle at center
def drawCircle(image, ellipset, ellipseu):
    xt, yt = ellipset[0]
    xu, yu = ellipseu[0]
    cv2.circle(image, (int(xt), int(yt)), 3, (255, 64, 128), -1)
    cv2.circle(image, (int(xu), int(yu)), 3, (128, 192, 255), -1)

# Caculate major radius
def calMradius(d1, d2, angle):
    rmajor = max(d1, d2)/2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    return rmajor, angle

# Draw line
def drawLine(image, xt, yt, xu, yu, anglet, angleu, rmajort, rmajoru):
    xtopt = xt + math.cos(math.radians(anglet))*rmajort
    ytopt = yt + math.sin(math.radians(anglet))*rmajort
    xbott = xt + math.cos(math.radians(anglet + 180))*rmajort
    ybott = yt + math.sin(math.radians(anglet + 180))*rmajort

    xtopu = xu + math.cos(math.radians(angleu))*rmajoru
    ytopu = yu + math.sin(math.radians(angleu))*rmajoru
    xbotu = xu + math.cos(math.radians(angleu + 180))*rmajoru
    ybotu = yu + math.sin(math.radians(angleu + 180))*rmajoru

    cv2.line(image, (int(xtopt), int(ytopt)), (int(xbott), int(ybott)), (255, 126, 0), 2)
    cv2.line(image, (int(xtopu), int(ytopu)), (int(xbotu), int(ybotu)), (0, 126, 255), 2)
    
# Put Text
def putText(image, dt, du):
    cv2.putText(image, 'TL/UL=%.3f' %(dt/du), (150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)



def linearEquation(dot1, dot2, img):
    [x1, y1] = list(dot1)
    [x2, y2] = list(dot2)

    a = 0
    b = 0
    c = 0

    if abs(x2 - x1) > abs(y2 - y1):
        if (x2 - x1)==0:
            b = -1
            a = 0
            c = -b * y1
        else:
            b = -1
            a = (y2 - y1) / (x2 - x1)
            c = -a * x1 - b * y1
    else:
        if (y2 - y1)==0:
            a = -1
            b = 0
            c = -a * x1
        else:
            a = -1
            b = (x2 - x1) / (y2 - y1)
            c = -a * x1 - b * y1
    
    dots = [[], []]
    if b == -1:
        minX = x1 if x1 < x2 else x2
        maxX = x1 if x1 > x2 else x2
        for x in range(0, minX):
            y = (-a * x - c) / b
            y = int(y)
            if y<len(img):
                if (img[y][x] > 0):
                    dots[0].append(x)
                    dots[0].append(y)
                    break
            else:
                y=(len(img)-1)
                if (img[y][x] > 0):
                    dots[0].append(x)
                    dots[0].append(y)
                    break

        for x in range(maxX, len(img[0])):
            y = (-a * x - c) / b
            y = int(y)
            if (img[y][x] == 0):
                dots[1].append(x - 1)
                dots[1].append(int((-a * (x - 1) - c) / b))
                break
    else:
        minY = y1 if y1 < y2 else y2
        maxY = y1 if y1 > y2 else y2
        for y in range(0, minY):
            x = (-b * y - c) / a
            x = int(x)
            if x<len(img[0]):
                if (img[y][x] > 0):
                    dots[0].append(x)
                    dots[0].append(y)
                    break
            else:
                x=(len(img[0])-1)
                if (img[y][x] > 0):
                    dots[0].append(x)
                    dots[0].append(y)
                    break
        for y in range(maxY, len(img)):
            x = (-b * y - c) / a
            x = int(x)
            if (img[y][x] == 0):
                dots[1].append(int((-b * (y - 1) - c) / a))
                dots[1].append(y - 1)
                break
    
    return dots

def getk(p0, p1):
    if p0[1]-p1[1]==0:
        Slopek="0"
    elif p0[0]-p1[0]==0:
        Slopek=0
    else:
        Slopek = (p0[1]-p1[1]) / (p0[0]-p1[0])
        Slopek=-(1/Slopek)

    return Slopek

def RatioLen(Pointk0,Pointk1,Pointk2):
    Pointk0 = list(Pointk0)
    Pointk1 = list(Pointk1)
    Pointk2 = list(Pointk2)
    L1 = getDistance(Pointk0,Pointk1)
    L2 = getDistance(Pointk0,Pointk2)
    if L2==0:
        ratio = 0
    else:
        ratio =L1/L2
    return ratio

def getDistance(pA, pB):
    p1 = np.array(pA)
    p2 = np.array(pB)
    p3 = p2 - p1
    p4 = math.hypot(p3[0], p3[1])
    return p4



def contourpoint(p1,p2,binary):
    
    [x1, y1] = list(p1)
    [x2, y2] = list(p2)
    conts, _ = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # 颜色区分，把原图再转为BGR
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(binary, conts, -1, (0, 255, 0), 1)
    x, y, w, h = cv2.boundingRect(conts[0])
    cv2.rectangle(binary, (x, y), (x + w, y + h), (255, 0, 0), 1)

    if x2 != x1:  # 若存在斜率 y=kx+b
        k = (y2 - y1) / (x2 - x1)
        b = y1 - x1 * k
        # 求解直线和boundingbox的交点A和B
        pa, pb = (x, int(k * x + b)), ((x + w), int(k * (x + w) + b))
    else:  # 若斜率不存在，垂直的直线
        pa, pb = (x1, y), (x1, y + h)
    points = []
    for pt in zip(*line(*pa, *pb)):
        if cv2.pointPolygonTest(conts[0], pt, False) == 0:  # 若点在轮廓上
            points.append(pt)

    return points



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

    big_contourt = fLarContour(binary)
    big_contouru = fLarContour(Ubinary)

    ellipset = cv2.fitEllipse(big_contourt)
    ellipseu = cv2.fitEllipse(big_contouru)
    (xt, yt), (dt1, dt2), anglet = ellipset
    (xu, yu), (du1, du2), angleu = ellipseu

    drawEllipse(imgresult, ellipset, ellipseu)

    # drawCircle(imgresult, ellipset, ellipseu)

    rmajort, anglet = calMradius(dt1, dt2, anglet)
    rmajoru, angleu = calMradius(du1, du2, angleu)

    #drawLine(imgresult, xt, yt, xu, yu, anglet, angleu, rmajort, rmajoru)

    xtopt = xt + math.cos(math.radians(anglet))*rmajort
    ytopt = yt + math.sin(math.radians(anglet))*rmajort
    xbott = xt + math.cos(math.radians(anglet + 180))*rmajort
    ybott = yt + math.sin(math.radians(anglet + 180))*rmajort

    xtopu = xu + math.cos(math.radians(angleu))*rmajoru
    ytopu = yu + math.sin(math.radians(angleu))*rmajoru
    xbotu = xu + math.cos(math.radians(angleu + 180))*rmajoru
    ybotu = yu + math.sin(math.radians(angleu + 180))*rmajoru

    #
    blank = np.zeros((512,512),np.uint8)
    cv2.line(blank,(int(xtopu), int(ytopu)),(int(xbotu), int(ybotu)),(255,0,0),1)

    piontslists = []
    col=len(blank[0])
    row=len(blank)
    for m in range(0,col):
        for j in range(0,row):
            if blank[m][j]== 255:
                piontslists.append([m,j]) 

    MaxRat = []
    MaxpointdotsT = []
    MaxpointdotsU = []
    Maxpoint = []
    pointsNum = len(piontslists[:])
    for k in range(3,pointsNum-3):
        # cv2.line(img,tuple(reversed(piontslists[k-3])),tuple(reversed(piontslists[k+3])),(0,255,0),1)
        Slopek = getk(piontslists[k-3],piontslists[k+3])
        k1=[[], []]
        if Slopek=="0":
            k1[0] = piontslists[k][0]
            k1[1] = piontslists[k][1]+500
        else:
            b = piontslists[k][1] - Slopek * piontslists[k][0]
            k1[0] = int(piontslists[k][0]+500)
            k1[1] = int(Slopek*k1[0] + b)
        
        
            
        cv2.line(imgresult,(int(xtopu), int(ytopu)),(int(xbotu), int(ybotu)),(255,0,0),2)

        dotsU = contourpoint(reversed(piontslists[k]),reversed(k1),Ubinary)
        dotsT = contourpoint(reversed(piontslists[k]),reversed(k1),binary)
        if len(dotsT) == 0 or len(dotsU) == 0:
            dotsU = linearEquation(reversed(piontslists[k]),reversed(k1),Ubinary)
            dotsT = linearEquation(reversed(piontslists[k]),reversed(k1),binary)
            if len(dotsT) == 2 and len(dotsU) == 2 :      
                if dotsU[0]==[] and dotsT[0]==[] and dotsU[1]!=[] and dotsT[1]!=[] : 
                    T1 = list(dotsT[1])
                    U1 = list(dotsU[1])
                    piontsk = list(reversed(piontslists[k]))
                    D3 = getDistance(T1,U1)
                    D4 = getDistance(piontsk,U1)
                    if D3 <D4:               
                        R1 = RatioLen(reversed(piontslists[k]),dotsT[1],dotsU[1]) 
                        if R1<1 :          
                            MaxRat.append(R1)
                            MaxpointdotsT.append(dotsT[1])
                            MaxpointdotsU.append(dotsU[1])
                            Maxpoint.append(piontslists[k])
                elif dotsU[1]==[] and dotsT[1]==[] and dotsU[0]!=[] and dotsT[0]!=[]:
                    T0 = list(dotsT[0])
                    U0 = list(dotsU[0])
                    piontsk = list(reversed(piontslists[k]))
                    D1 = getDistance(T0,U0)
                    D2 = getDistance(piontsk,U0)
                    if D1<D2:
                        R0 = RatioLen(reversed(piontslists[k]),dotsT[0],dotsU[0]) 
                        if R0 < 1:           
                            MaxRat.append(R0)
                            MaxpointdotsT.append(dotsT[0])
                            MaxpointdotsU.append(dotsU[0])
                            Maxpoint.append(piontslists[k])
                elif dotsU[1]!=[] and dotsT[1]!=[] and dotsU[0]!=[] and dotsT[0]!=[]:
                    T0 = list(dotsT[0])
                    U0 = list(dotsU[0])
                    T1 = list(dotsT[1])
                    U1 = list(dotsU[1])
                    piontsk = list(reversed(piontslists[k]))
                    D1 = getDistance(T0,U0)
                    D2 = getDistance(piontsk,U0)
                    D3 = getDistance(T1,U1)
                    D4 = getDistance(piontsk,U1)
                    # D5 = getDistance(piontsk,T0)
                    # D6 = getDistance(piontsk,T1)   and D2 > D5 and D4 > D6
                    if D1 < D2 and D3 < D4:            
                        R0 = RatioLen(reversed(piontslists[k]),dotsT[0],dotsU[0])
                        R1 = RatioLen(reversed(piontslists[k]),dotsT[1],dotsU[1])
                        if R0>R1:
                            MaxRat.append(R0)
                            MaxpointdotsT.append(dotsT[0])
                            MaxpointdotsU.append(dotsU[0])
                            Maxpoint.append(piontslists[k])
                            
                        else:
                            MaxRat.append(R1)
                            MaxpointdotsT.append(dotsT[1])
                            MaxpointdotsU.append(dotsU[1])
                            Maxpoint.append(piontslists[k])
                    elif D1 > D2 and D3 < D4:
                        R0 = 0
                        R1 = RatioLen(reversed(piontslists[k]),dotsT[1],dotsU[1])
                        if R0>R1:
                            MaxRat.append(R0)
                            MaxpointdotsT.append(dotsT[0])
                            MaxpointdotsU.append(dotsU[0])
                            Maxpoint.append(piontslists[k])
                            
                        else:
                            MaxRat.append(R1)
                            MaxpointdotsT.append(dotsT[1])
                            MaxpointdotsU.append(dotsU[1])
                            Maxpoint.append(piontslists[k])
                    elif D1 < D2 and D3 > D4:
                        R0 = RatioLen(reversed(piontslists[k]),dotsT[0],dotsU[0])
                        R1 = 0
                        if R0>R1:
                            MaxRat.append(R0)
                            MaxpointdotsT.append(dotsT[0])
                            MaxpointdotsU.append(dotsU[0])
                            Maxpoint.append(piontslists[k])
                            
                        else:
                            MaxRat.append(R1)
                            MaxpointdotsT.append(dotsT[1])
                            MaxpointdotsU.append(dotsU[1])
                            Maxpoint.append(piontslists[k])



     
        
        elif len(dotsT) == 2 and len(dotsU) == 2  :
            T0 = list(dotsT[0])
            U0 = list(dotsU[0])
            T1 = list(dotsT[1])
            U1 = list(dotsU[1])
            piontsk = list(reversed(piontslists[k]))
            D1 = getDistance(T0,U0)
            D2 = getDistance(piontsk,U0)
            D3 = getDistance(T1,U1)
            D4 = getDistance(piontsk,U1)
            # D5 = getDistance(piontsk,T0)
            # D6 = getDistance(piontsk,T1)   and D2 > D5 and D4 > D6
            if D1 < D2 and D3 < D4:            
                R0 = RatioLen(reversed(piontslists[k]),dotsT[0],dotsU[0])
                R1 = RatioLen(reversed(piontslists[k]),dotsT[1],dotsU[1])
                if R0>R1:
                    MaxRat.append(R0)
                    MaxpointdotsT.append(dotsT[0])
                    MaxpointdotsU.append(dotsU[0])
                    Maxpoint.append(piontslists[k])
                    
                else:
                    MaxRat.append(R1)
                    MaxpointdotsT.append(dotsT[1])
                    MaxpointdotsU.append(dotsU[1])
                    Maxpoint.append(piontslists[k])
            elif D1 > D2 and D3 < D4:
                R0 = 0
                R1 = RatioLen(reversed(piontslists[k]),dotsT[1],dotsU[1])
                if R0>R1:
                    MaxRat.append(R0)
                    MaxpointdotsT.append(dotsT[0])
                    MaxpointdotsU.append(dotsU[0])
                    Maxpoint.append(piontslists[k])
                    
                else:
                    MaxRat.append(R1)
                    MaxpointdotsT.append(dotsT[1])
                    MaxpointdotsU.append(dotsU[1])
                    Maxpoint.append(piontslists[k])
            elif D1 < D2 and D3 > D4:
                R0 = RatioLen(reversed(piontslists[k]),dotsT[0],dotsU[0])
                R1 = 0
                if R0>R1:
                    MaxRat.append(R0)
                    MaxpointdotsT.append(dotsT[0])
                    MaxpointdotsU.append(dotsU[0])
                    Maxpoint.append(piontslists[k])
                    
                else:
                    MaxRat.append(R1)
                    MaxpointdotsT.append(dotsT[1])
                    MaxpointdotsU.append(dotsU[1])
                    Maxpoint.append(piontslists[k])
            
            
    if len(MaxRat)>0:
        
        cv2.line(imgresult,tuple(reversed(Maxpoint[MaxRat.index(max(MaxRat))])),tuple(MaxpointdotsT[MaxRat.index(max(MaxRat))]),(0,255,0),2)
        cv2.line(imgresult,tuple(reversed(Maxpoint[MaxRat.index(max(MaxRat))])),tuple(MaxpointdotsU[MaxRat.index(max(MaxRat))]),(0,255,0),2)
        cv2.circle(imgresult,tuple(MaxpointdotsT[MaxRat.index(max(MaxRat))]),3, (255, 0, 0), -1)
        cv2.circle(imgresult,tuple(MaxpointdotsU[MaxRat.index(max(MaxRat))]),3, (255, 0, 0), -1) 
        cv2.putText(imgresult, 'Ratio=%.3f' %(max(MaxRat)), (150, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
        ratio = '-Ratio=%.3f' %(max(MaxRat))         
        cv2.imwrite(save_path + file + ratio + "-2-1.png", imgresult)
    else:
        cv2.imwrite(save_path + file + ratio + "-2-0.png", imgresult)

    
print('success process')