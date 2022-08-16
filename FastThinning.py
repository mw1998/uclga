import matplotlib.pyplot as plt 
import numpy as np
import os
import skimage.io as io
from skimage import filters
from skimage.morphology import disk
from PIL import Image
from cv2 import cv2
import math
from skimage.draw import line



source_path = r'TestImage/'
save_path = r'TestFTResults/'


 
def neighbours(x,y,image):
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [img[x_1][y],img[x_1][y1],img[x][y1],img[x1][y1],         # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9
 
# 计算邻域像素从0变化到1的次数
def transitions(neighbours):
    n = neighbours + neighbours[0:1]      # P2,P3,...,P8,P9,P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)
 
# Zhang-Suen 细化算法
def zhangSuen(image):
    Image_Thinned = image.copy()  # Making copy to protect original image
    changing1 = changing2 = 1
    while changing1 or changing2:   # Iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned

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

def getk(p0, p1):
    if p1[1]-p0[1]==0:
        Slopek= "0"
    elif p1[0]-p0[0]==0:
        Slopek=0
    else:
        Slopek = (p0[1]-p1[1]) / (p0[0]-p1[0])
        Slopek=-(1/Slopek)

    return Slopek

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


def Search_curve_endpoint(image) :
    
    img_reverse = image.copy() # Get a image matrix from total dataset

    #img_reverse = np.where(img_reverse > 0, 255 , 0)  # Convert the binarized picture into the picture with black pixels on a white background
    #cv2.imwrite("E:\\img\\{}.png".format(str(img_number)),img_reverse) # Save the picture with black pixels on a white background

    #print("----第",img_number,"图----")

    img_reverse = cv2.bitwise_xor(img_reverse , 1)  # Perform the exclusive OR operation of all pixels in the picture . Replace two values with each other  0^1=1 1^1=0
    #cv2.imshow("reserve", img_reverse )#
    #print(img_reverse .shape)#

    (rows, cols) = np.nonzero(img_reverse)# Take out the non-zero point index, that is, the coordinates of all target pixels
    for i in range(len(rows)):
        if rows[i] == len(image)-1:
            rows[i] -= 1
    for j in range(len(cols)):
        if cols[j] == len(image[0])-1:
            cols[j] -= 1

    Max_pixel_number = len(rows)# Calculate the number of target pixels
    #print("----本图共有",Max_pixel_number,"个点----")

    skel_coords = [] # Store the end coordinates of the skeleton
    Total_pixel = [] # Store all unsorted target pixel's coordinates

    for (r, c) in zip(rows, cols):
        Total_pixel.append((r, c)) # Store unsorted target pixel's coordinates
        (col_neigh, row_neigh) = np.meshgrid(np.array([c - 1, c, c + 1]), np.array([r - 1, r, r + 1])) # Take the target pixel as the center to obtain the nine-square grid
        col_neigh = col_neigh.astype('int')#Convert to integer
        row_neigh = row_neigh.astype('int')#Convert to integer
        img_filter = img_reverse [row_neigh, col_neigh].ravel() != 0 # 1  converted to true and 0  converted to false
        #print('坐标点：(', r, ",", c, ')', "九宫格：", img_reverse [row_neigh, col_neigh].ravel())

        n1r = img_filter[0] + img_filter[1] + img_filter[2] # The top side of the Nine-square grid of the nine-square grid which centered on the target pixel
        n3r = img_filter[6] + img_filter[7] + img_filter[8] # The botton side of the Nine-square grid of the nine-square grid which centered on the target pixel
        n1c = img_filter[0] + img_filter[3] + img_filter[6] # The left side of the Nine-square grid of the nine-square grid which centered on the target pixel
        n3c = img_filter[2] + img_filter[5] + img_filter[8] # The right side of the Nine-square grid of the nine-square grid which centered on the target pixel
        n2r = img_filter[1] + img_filter[7]
        n2c = img_filter[3] + img_filter[5]

        if ((2 <= np.sum(img_filter != 0) <= 3) ) & (
                (n1r + n1c == 0) | (n1r + n3c == 0) | (n3r + n1c == 0) | (n3r + n3c == 0))&((n2r!=n2c)|(n2r==n2c==0)):
            skel_coords.append((r, c))#Storage endpoint
    #print("图像端点：",skel_coords,"\n\n")
    return [img_reverse, skel_coords , Total_pixel,Max_pixel_number]

# **********************************************************************************************************************
# FUNCTION  ：Input the coordinate of two endpoints and get the order coordinate of pixels
# PARAMETER ：A list include : [Image list after reverse pixels , The coordinate of two endpoint , The list of all pixels found , The number of all pixels found]
# RETURN    : The order coordinate list of pixels   If there is an error , return zero
# **********************************************************************************************************************

def Print_pixel_coords(img_reverse,Endpoint_coords,Total_pixel,Max_pixel_number) :
    Pixel_coords = []  # Store the sorted coordinates of all pixels
    Endpoint_num = len(Endpoint_coords) # Get the number of endpoints

    if Endpoint_num == 2: # If the number of  endpoints is not two , it will must be error
       
        Start_point = list(Endpoint_coords[0]) # The coordinate of the start endpoint
        End_point = list(Endpoint_coords[1]) # The coordinate of the end endpoint
        #print("起始点：",Start_point)
        x = Start_point[0] # The abscissa of the start endpoint
        y = Start_point[1] # The Ordinate of the start endpoint
        pixel_number = 0 # 累计遍历的像素点个数

        while True:
            Pixel_coords.append((x, y)) # Store pixels in order

            pixel_number = pixel_number+1
            if pixel_number>(Max_pixel_number) :
               # print("结果出错，坐标点数据量已经溢出")
                break # The result is  error, the number of coordinate point data has overflowed
            if (x, y) not in Total_pixel :
               # print("结果出错，数据结果不在库中")
                break # The result is wrong, the data result is not in the library

            (y_neigh, x_neigh) = np.meshgrid(np.array([y - 1, y, y + 1]), np.array([x - 1, x, x + 1])) # Take the target pixel as the center to obtain the nine-square grid

            x_neigh = x_neigh.astype('int') #Convert to integer
            y_neigh = y_neigh.astype('int') #Convert to integer

            Next_point = img_reverse [x_neigh, y_neigh].ravel() != 0
            #print('第',pixel_number,'点：(', x, ",", y,")")#, ')', "九宫格：", img_reverse[x_neigh, y_neigh].ravel())

            p_cross = Next_point[1] + Next_point[3] + Next_point[5] + Next_point[7] # The top, bottom, left, and right cross areas of the nine-square grid which centered on the target pixel
            p_four_corners = Next_point[0] + Next_point[2] + Next_point[6] + Next_point[8] # Four corners of the nine-square grid which centered on the target pixel

            if p_cross != 0: # The next target pixel is detected in the cross area
                img_reverse[x, y] = 0  # Erase this pixel
                if Next_point[1] == 1:
                    x -= 1
                if Next_point[7] == 1:
                    x += 1
                if Next_point[3] == 1:
                    y -= 1
                if Next_point[5] == 1:
                    y += 1
                # print("十字，下一个中心点：",x,y)
            else :
                img_reverse[x, y] = 0 # Erase this pixel
                if p_four_corners == 0 : # In addition to this pixel, if there is empty in the nine-square grid, an error will be reported
                  # print("！****！错误，未搜索到下一点!****！")
                  break
                if Next_point[0] == 1:
                    x -= 1
                    y -= 1
                if Next_point[6] == 1:
                    x += 1
                    y -= 1
                if Next_point[2] == 1:
                    x -= 1
                    y += 1
                if Next_point[8] == 1:
                    x += 1
                    y += 1
                #print("四角，下一个中心点：", x,y)

            if (x, y) == Endpoint_coords[1]: # Determine whether the current pixel is another endpoint
                #print("结束点：[",x,',',y,"]\n\n\n\n")
                Pixel_coords.append((x, y)) #Store the last endpoint
                #print("初始获取的像素点坐标：",Total_pixel)
                #print("顺序排序后像素点坐标：",Pixel_coords)
                #img_reverse = np.where(img_reverse > 0, 0, 255)
                #cv2.imwrite("E:\\img2\\{}-{}.png".format(str(img_number), str(pixel_number)), img_reverse)
                #print("打印：", img_number, "-", pixel_number)
                #img_reverse = np.where(img_reverse > 0, 0, 1)
                if pixel_number+1 < (Max_pixel_number): # Determine whether the traversed pixels are less than the total number of pixels
                      # The data result is wrong, the coordinate  data is missing
                    print(Total_pixel)
                    print(pixel_number)
                break
    return Pixel_coords
            #img_reverse = np.where(img_reverse > 0, 0, 255)
            #cv2.imwrite("E:\\img2\\{}-{}.png".format(str(img_number), str(pixel_number)), img_reverse)
            #print("打印：", img_number, "-", pixel_number)
            #img_reverse = np.where(img_reverse > 0, 0, 1)


def contourpoint(p1,p2,binary):
    
    [x1, y1] = list(p1)
    [x2, y2] = list(p2)
    conts, _ = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # 颜色区分，把原图再转为BGR
    if len(conts) != 0:
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
    imgresult = img.copy()   
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_original = filters.median(imgGray, disk(5))
    tumorBinary = img_original.copy()
    tumorBinary = img_original != 0
    tumorSkeleton = zhangSuen(tumorBinary)
    tumorSkeleton = np.invert(tumorSkeleton)
    uterskeleton = np.array(tumorSkeleton).astype('uint8')
    
    
    Order = Search_curve_endpoint(uterskeleton)
    uterskeleton = Print_pixel_coords(Order[0],Order[1],Order[2],Order[3])
    piontslists = []
    for point in uterskeleton:
        piontslists.append(point)

    
    for p in piontslists[:]:
        cv2.circle(imgresult,tuple(reversed(p)),1, (255, 0, 0), -1)
    # cv2.imshow("img1",imgresult)
    # cv2.waitKey(0)
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imggray = gray.copy()
    thresh = tumorbinary(imggray)
    # thresh = np.invert(thresh)
    Ubinary = uterusbinary(gray)

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
            
        # dotsU = contourpoint(reversed(piontslists[k]),reversed(k1),Ubinary)
        # dotsT = contourpoint(reversed(piontslists[k]),reversed(k1),thresh)

        dotsU = contourpoint(reversed(piontslists[k]),reversed(k1),Ubinary)
        dotsT = contourpoint(reversed(piontslists[k]),reversed(k1),thresh)

        # dotsU = linearEquation(reversed(piontslists[k]),reversed(k1),Ubinary)
        # dotsT = linearEquation(reversed(piontslists[k]),reversed(k1),thresh)
        if len(dotsT) == 0 or len(dotsU) == 0:
            dotsU = linearEquation(reversed(piontslists[k]),reversed(k1),Ubinary)
            dotsT = linearEquation(reversed(piontslists[k]),reversed(k1),thresh)
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
        elif len(dotsT) == 2 and len(dotsU) == 2:
            T0 = list(dotsT[0])
            U0 = list(dotsU[0])
            T1 = list(dotsT[1])
            U1 = list(dotsU[1])
            piontsk = list(reversed(piontslists[k]))
            D1 = getDistance(T0,U0)
            D2 = getDistance(piontsk,U0)
            D3 = getDistance(T1,U1)
            D4 = getDistance(piontsk,U1)
            D5 = getDistance(piontsk,T0)
            D6 = getDistance(piontsk,T1)
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
        cv2.putText(imgresult, 'Ratio=%.3f' %(max(MaxRat)), (150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)      
        ratio = '-Ratio=%.3f' %(max(MaxRat))       
        cv2.imwrite(save_path + file + ratio + "-3-1.png", imgresult)
    else:
        cv2.imwrite(save_path + file + "-3-0.png", imgresult)

    # cv2.imshow("img1",imgresult)
    # cv2.waitKey(0)
      


print('success process')

