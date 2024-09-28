from pyparma import *
from z3 import *
import random
from time import sleep
import environment
import hashlib
import math
import time
from time import time
from datetime import datetime 
# import dnn
import threading
import pyparma_posInvRegion40
# import pyparma_posInvRegion39
# import pythonRenderAnImage2
# import singleTriangelInvRegionZ3_2
# import eran_master.tf_verify.interval_image_translator_2habeeb as deepPoly
# import interval_image_translator_3habeeb
import anytree
import os
import sys
import itertools



import ast

from collections import Counter

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2 
from tensorflow.keras.models import load_model

#import gurobiGetDepths2
# import gurobiGetDepths4
import generateVnnlbPropertyfile
import intervalImageFunctions1
# import singleTriangleFunctions1

import oldInvComputation1
# import oldInvRegionSolver

import cv2
import onnx
import onnxruntime

from onnx import numpy_helper

posZp100 = -1000 


testPosXp =0
testPosYp =4.5
testPosZp = 194.5

globalIntervalImage = {}
finalGlobalIntervalImage = dict()
allGlobalIntervalImages = []
currTriangleUniquePositions = 0
edges = []

no_of_vars = 100

pVars = []
aVars = []
bVars = []
cVars = []
dVars = []

for iii in range(no_of_vars):
    pVars += [Real('p%d' % iii)]
    aVars += [Real('a%d' % iii)]
    bVars += [Real('b%d' % iii)]
    cVars += [Real('c%d' % iii)]
    dVars += [Real('d%d' % iii)]



stopRoundingFlag =0
stopDownRoundingFlag =0
currentImageRepeatCount = 0

globalCurrentImage = []
globalInsideVertexDataToPPL =[]
globalIntersectingVertexDataToPPL = []

dataToComputeIntervalImage = dict()
currTriangleIntervalImage = dict()
dictionaryOfTriangleIntervalImages = dict()


allImages = []
vertices = environment.vertices
numOfVertices = environment.numOfVertices
tedges = environment.tedges
numOftedges = environment.numOfEdges
nvertices = environment.nvertices

imageWidth = environment.imageWidth
imageHeight = environment.imageHeight

# intiFrusCons = environment.intiFrusCons
# initCubeCon = environment.initCubeCon
# x0 = environment.x0
# x1 = environment.x1
# y0 = environment.y0
# y1 = environment.y1
# zmin = environment.z0
# zmax = environment.z1
canvasWidth = environment.canvasWidth
canvasHeight = environment.canvasHeight
focalLength = environment.focalLength
t= environment.t
b = environment.b
l = environment.l
r = environment.r
n = environment.n
f = environment.f

outValues = [0]*numOfVertices*4*5


xp0, yp0, zp0 = Reals('xp0 yp0 zp0')
# xp1, yp1, zp1 = Reals('xp1 yp1 zp1')

p0,q0 = Reals('p0 q0')
u0, v0, w0 = Reals('u0 v0 w0')

p1,q1 = Reals('p1 q1')
u1, v1, w1 = Reals('u1 v1 w1')

p2,q2 = Reals('p2 q2')
u2, v2, w2 = Reals('u2 v2 w2')

p3,q3 = Reals('p3 q3')
u3, v3, w3 = Reals('u3 v3 w3')

numOfZ3Variables = 300
currZ3VariableIndex = 0
# Int('A_{0}'.format(idx)) for idx in range(0, A_length)
p_list = [Real('p{}'.format(i)) for i in range(0,numOfZ3Variables)]
q_list = [Real('q{}'.format(i)) for i in range(0,numOfZ3Variables)]
u_list = [Real('u{}'.format(i)) for i in range(0,numOfZ3Variables)]
v_list = [Real('v{}'.format(i)) for i in range(0,numOfZ3Variables)]
w_list = [Real('w{}'.format(i)) for i in range(0,numOfZ3Variables)]
g_list = [Real('g{}'.format(i)) for i in range(0,numOfZ3Variables)]
xl_list = [Real('xl{}'.format(i)) for i in range(0,numOfZ3Variables)]
xk_list = [Real('xk{}'.format(i)) for i in range(0,numOfZ3Variables)]
yl_list = [Real('yl{}'.format(i)) for i in range(0,numOfZ3Variables)]
yk_list = [Real('yk{}'.format(i)) for i in range(0,numOfZ3Variables)]
zl_list = [Real('zl{}'.format(i)) for i in range(0,numOfZ3Variables)]
zk_list = [Real('zk{}'.format(i)) for i in range(0,numOfZ3Variables)]

# OpenGL perspective projection matrix
mProj = [
        [2 * n / (r - l), 0, 0, 0],
        [0,2 * n / (t - b),0,0],
        [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1 ],
        [0,0,-2 * f * n / (f - n),0]
    ]


def getDNNOutput_onnx(inputImage):
    
    # print("dnn started")
    # print(str(datetime.now()))
    model = onnx.load('OGmodel_pb_converted.onnx')


    dnnOutput = 1

    image = cv2.imread(inputImage)    
    image = cv2.resize(image, (49, 49)).copy()

    a, b, c = image.shape
    image = image.reshape(1, a, b, c)
    # print(image.shape)
    # print(image[0][0])

    image = image.astype(np.float32)  / 255.0
    # image2 = tf.convert_to_tensor(image)

    session = onnxruntime.InferenceSession('OGmodel_pb_converted.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # print(input_name)
    # print(output_name)

    result = session.run([output_name], {input_name: image})
    print(result)
    dnnOutput  = np.argmax(np.array(result).squeeze(), axis=0)

    # print("dnnOutput = ", dnnOutput)
    return dnnOutput 

def updateGlobalIntervalImage(numOfCurrInvRegions):
    # print("Update global image")
    singleTDataFile = open('singleTrianglePixelDatafromcpp.txt', 'r') 
    # singleTDataFile = open(fileName, 'r') 
    
    line = singleTDataFile.readline()
    
    backGroundData = [1, 25, 24, 0, 1000]
    # backGroundData = [230, 240, 250, 0, 1000]
    while line:
        
        # print("currentPixel = ", line)
        
        currentPixel = int(line)
        line = singleTDataFile.readline()
        numOfColours = int(line)
        
        currentPixelTrColors = []
        for i in range(0, numOfColours):
            
            #read each color details and create the interval for the current triangle.
            line = singleTDataFile.readline()
            colData = line.replace("\n","").split(",")
            currentPixelTrColors.append(colData)
            # print(colData)
        
        
        # print(currentPixelTrColors)
        #take union of these colours as interval
        
        #TODO:important, use set and the currTriangelUniquePositions
        # print("numOfColours, numOfCurrInvRegions, currTriangleUniquePositions ===> ",numOfColours, numOfCurrInvRegions, currTriangleUniquePositions)
        # if(numOfColours < numOfCurrInvRegions):
        #     currentPixelTrColors.append(backGroundData)
        
        if(numOfColours < currTriangleUniquePositions):
            currentPixelTrColors.append(backGroundData)
            # print("Adding background color ", numOfCurrInvRegions,", ",numOfColours,", ",\
            #     currentPixel,", ", currTriangleUniquePositions)
            # sleep(2)
        
        # print(currentPixelTrColors) 
            
        currPixelInterval = [0]*8
        
        for k in range(0,3):
            
            kth_elements = [sublist[k] for sublist in currentPixelTrColors]
            
            kth_elements = [float(x) for x in kth_elements]
            
            currPixelInterval[2*k+0] = min(kth_elements)
            currPixelInterval[2*k+1] = max(kth_elements)
        
        minD_elements = [sublist[3] for sublist in currentPixelTrColors]
        minD_elements = [float(x) for x in minD_elements]
        currPixelInterval[6] = min(minD_elements)
         
        maxD_elements = [sublist[4] for sublist in currentPixelTrColors]
        max_elements = [float(x) for x in maxD_elements]
        currPixelInterval[7] = max(max_elements)
         
        # print("currPixelInterval ==>",currPixelInterval)        
        # print("\n\n\n")
        
        # if(currentPixel == 1451):
        #     print("pixel 1451")
        #     # sleep(3)
        
        #read current interval data from the global interval images.
        
        if currentPixel in globalIntervalImage:
            # print("currentPixel present in the global image")
            currentIntValue = globalIntervalImage[currentPixel]
            
            # if(currentPixel == 1324):
            #     print("1324 currentPixel present in the global image")
            #     print("current global value = ", currentIntValue)
            #     print("Current triangle value = ", currPixelInterval)
            #     sleep(10)
            ##################overapproximation########TOREMOVE#######
            
            # tempList = []
            # tempList.append(currentIntValue)
            # tempList.append(currPixelInterval)
            
            # tempIntervals = [0]*8
            
            # for k in range(0,4):                
            #     kth_elements = [sublist[2*k+0] for sublist in tempList]
            #     k1th_elements = [sublist[2*k+1] for sublist in tempList]
                
            #     kth_elements = [float(x) for x in kth_elements]
            #     k1th_elements = [float(x) for x in k1th_elements]
                
            #     tempIntervals[2*k+0] = min(kth_elements)
            #     tempIntervals[2*k+1] = max(k1th_elements)
            
            # globalIntervalImage[currentPixel] = tempIntervals   
            
            ##################overapproximation###############
            
            #if current global depth's min is greater than the current pixel max depth then
            #assign the current triangle's interval as the global value.
            #if the current triangle's intervals min depth is greater than the global max depth
            # then do nothing           
            #else overlapping
            if currentIntValue[6] > currPixelInterval[7]:
                globalIntervalImage[currentPixel] = currPixelInterval
                
                # if(currentPixel == 1451):
                #     print("1451 Updated global interval with current pixel")
                #     print("new global value = ",  globalIntervalImage[currentPixel])
                
            elif currPixelInterval[6] > currentIntValue[7] :
                # if(currentPixel == 1451):
                #     print("1451 passed")
                pass
            else:
                
                # if(currentPixel == 1451):
                #     print("1451 extended global interval with current pixel")
                
                
                tempList = []
                tempList.append(currentIntValue)
                tempList.append(currPixelInterval)
                
                tempIntervals = [0]*8
                
                for k in range(0,4):                
                    kth_elements = [sublist[2*k+0] for sublist in tempList]
                    k1th_elements = [sublist[2*k+1] for sublist in tempList]
                    
                    kth_elements = [float(x) for x in kth_elements]
                    k1th_elements = [float(x) for x in k1th_elements]
                    
                    tempIntervals[2*k+0] = min(kth_elements)
                    tempIntervals[2*k+1] = max(k1th_elements)
                
                globalIntervalImage[currentPixel] = tempIntervals          
        else:
            # if(currentPixel == 1324):
            #     print("1324 currentPixel not present in the global image")
            #     print("update global with ", currPixelInterval)
            #     sleep(2)
            
            globalIntervalImage[currentPixel] = currPixelInterval     
            
            
        
        line = singleTDataFile.readline()



def updateGlobalIntervalImage2(currTriangle,numOfRegions):
    # print("\n###########updateGlobalIntervalImage2###########")
    currTrIntImage = dictionaryOfTriangleIntervalImages[currTriangle]
    
    # print("currTrIntImage = ", currTrIntImage)
    # print("global IntervalImage = ", globalIntervalImage)
    
    for currPixel, currIntervals in currTrIntImage.items():
        # print(currPixel, ": ", currIntervals)
        if globalIntervalImage.get(currPixel):
            # print("pixel present in global image")
            currData = globalIntervalImage[currPixel]
            if(currIntervals[6] < currData[0][0]):
                globalIntervalImage[currPixel][0][0] = min(currData[0][0], currIntervals[7])
                globalIntervalImage[currPixel].append(currIntervals)
                # print("updated global interval image = ", currPixel, globalIntervalImage[currPixel])
                # sleep(4)
            
        else:
            # print("pixel not present in global image")
            
            
            globalIntervalImage[currPixel] = [[currIntervals[7]]]
            globalIntervalImage[currPixel].append(currIntervals)
    
    
    



def getDNNOutput(inputImage):
    model = load_model('saved_models/3_2')
    # print("inputImage = ",inputImage)
    dnnOutput = 1

    image = cv2.imread(inputImage)    
    image = cv2.resize(image, (49,49)).copy()

    a,b,c = image.shape
    image = image.reshape(1,a,b,c)
    image = image.astype(np.float32) / 255.0
    image2 = tf.convert_to_tensor(image)
    dnnOutput  = np.argmax(model.predict(image2))

    # print("dnnOutput = ", dnnOutput)
    return dnnOutput 






##############onnx #############

# import onnx
# import onnxruntime
# import cv2 
# from tensorflow.keras.models import load_model
# import tensorflow as tf

# model_onnx = onnx.load('iisc_net1.onnx')
# onnx.checker.check_model(model_onnx)
def getDNNOutput_onnx(inputImage,networkName):
    # dnnOutput = 1
    # model = onnx.load(networkName)

    image = cv2.imread(inputImage)  
    # print(image.shape)  
    image = cv2.resize(image, (49, 49)).copy()
    np.set_printoptions(threshold=sys.maxsize)
    # print("\n\n")
    
    # print(image)
    
    # print("\n------------\n")

    if networkName == "iisc_net1.onnx":
        a, b, c = image.shape
        image = image.reshape(1, c,b,a)
        # print(image.shape)
    else:
        a, b, c = image.shape
        # print(a,b,c)
        # exit()
        image = image.reshape(1, a,b,c)
        # print(image.shape)
        # 

    image = image.astype(np.float32) / 255.0
    # image2 = tf.convert_to_tensor(image)
    
    # print("\n\n")
    
    # print(image)
    
    # print("\n\n")
    
    # exit()
    

    session = onnxruntime.InferenceSession(networkName)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # print(input_name)
    # print(output_name)

    result = session.run([output_name], {input_name: image})
    # print("result = ", result)
    dnnOutput  = np.argmax(np.array(result).squeeze(), axis=0)
    # print("networkName = ", networkName)
    # print("dnnOutput = ", dnnOutput)
    return dnnOutput 








def planeEdgeIntersectionPython(planeId,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq):
    
    global outValues
   
    x1 = outValues[insideVertex*4+0]
    y1 = outValues[insideVertex*4+1]
    z1 = outValues[insideVertex*4+2]
    w1 = outValues[insideVertex*4+3]

    x2 = outValues[outsideVertex*4+0]
    y2 = outValues[outsideVertex*4+1]
    z2 = outValues[outsideVertex*4+2]
    w2 = outValues[outsideVertex*4+3]
    
    
    t1 = 0
    
    # print("x1,y1,z1,w1,x2,y2,z2,w2 ==> ",x1,y1,z1,w1,x2,y2,z2,w2)
    
    if(planeId == 3):
        t1= (-w1-x1)/(w2-w1+x2-x1)
    elif(planeId == 2):
        t1= (x1-w1)/(w2-w1-x2+x1)
        
    elif(planeId == 1):
        t1 = (-w1-y1)/(w2-w1+y2-y1)
    elif(planeId == 0):
        t1 = (-w1+y1)/(w2-w1-y2+y1)
    elif(planeId == 5):
        t1 = (-1-w1)/(w2-w1)
    elif(planeId ==4 ):
        t1 = (-1000-w1)/(w2-w1)
        
    # print("python intersection point ratio t1 = ",t1)
        
    intersectionPoint[0] = x1+t1*(x2-x1);
    intersectionPoint[1] = y1+ t1*(y2-y1);
    intersectionPoint[2] = z1+t1*(z2-z1);
    intersectionPoint[3] = w1+t1*(w2-w1)
    
    # print("intersection points  (before division)= ",intersectionPoint)
    
    temp0 = intersectionPoint[0]/intersectionPoint[3]
    temp1 = intersectionPoint[1]/intersectionPoint[3]
    # print("temp0(t0) ==> ",temp0," temp1(t1) ==> ", temp1)
    vertexPixelValue2[0] = min(environment.imageWidth-1, math.floor(((temp0 + 1) * 0.5 * 49)))
    vertexPixelValue2[1] =  min(environment.imageHeight-1, math.floor(((1 - (temp1 + 1) * 0.5) * 49)))
    # print("(((temp0 + 1) * 0.5 * 49)) ==>",(((temp0 + 1) * 0.5 * 49)))
    # print("Intersection point pixel value = ", vertexPixelValue2)
    
    # intersectionPoint.x = x1+t1*(x2-x1);
    # intersectionPoint.y = y1+ t1*(y2-y1);
    # intersectionPoint.z = z1+t1*(z2-z1);
    # *intersectionPointW = w1+t1*(w2-w1);
    
    
    # print("t1 = ",t1)
    # xv0 = vertices[insideVertex*3+0] - posXp
    # yv0 = vertices[insideVertex*3+1] - posYp
    # zv0 = vertices[insideVertex*3+2] - posZp
    # wv0 = -(vertices[insideVertex*3+2] - posZp)
    
    # xv1 = vertices[outsideVertex*3+0] - posXp
    # yv1 = vertices[outsideVertex*3+1] - posYp
    # zv1 = vertices[outsideVertex*3+2] - posZp
    # wv1 =  - (vertices[outsideVertex*3+2] - posZp)
    
    xv0 = vertices[insideVertex*3+0] 
    yv0 = vertices[insideVertex*3+1] 
    zv0 = vertices[insideVertex*3+2] 
    wv0 = -(vertices[insideVertex*3+2] )
    
    xv1 = vertices[outsideVertex*3+0] 
    yv1 = vertices[outsideVertex*3+1] 
    zv1 = vertices[outsideVertex*3+2] 
    wv1 =  - (vertices[outsideVertex*3+2] )
    
    outsideFraction = t1
    
    intersectionPoint[0] =(1- outsideFraction)*xv0+ outsideFraction*xv1 
    intersectionPoint[1] = (1- outsideFraction)*yv0+ outsideFraction*yv1 
    intersectionPoint[2] = (1- outsideFraction)*zv0+ outsideFraction*zv1 
    intersectionPoint[3] = (1- outsideFraction)*wv0+ outsideFraction*wv1 
    # print("intersection points = ",intersectionPoint)
    
    
        







    
def getVertexPixelValueIntersectZ3(x,y,z, plane):
    s3 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=True)
    # set_option(precision=20)
    set_param('parallel.enable', True)
    s3.set("sat.local_search_threads", 28)
    s3.set("sat.threads", 28)
    s3.set("sat.threads", 28)
    s3.check()
    
    a,b = Reals('a b')
    cons1 = ( (((-68.39567*(x ) )/ (z) )+ 24.5 ) == a )
    cons2 =  ( (((68.39567*(y ) )/ (z) )+ 24.5 ) == b )
    
    s3.add(simplify(And(cons1,cons2)))
    p = [0,0]
    if s3.check() == sat:
        m = s3.model()
        #print("etVertexPixelValueIntersectZ3: model from solver :",m)
        # a1 = str(eval("m[a].numerator_as_long()/m[a].denominator_as_long()"))
        # b1 = str(eval("m[b].numerator_as_long()/m[b].denominator_as_long()"))
        # # print(a1,b1)   
        # a1= float(a1)
        # b1= float(b1)
        
        a2 = str(eval("m[a]")).replace("?","")
        b2 = str(eval("m[b]")).replace("?","")
        # print("a2 = ", a2)
        # print("b2 = ", b2)
        
        # a2 = a2.split(".")[0]
        # b2 = b2.split(".")[0]
        # print("a2 = ", a2)
        # print("b2 = ", b2)
        
        if "." in a2:
            ta1 = int(a2.split(".")[0])
            ta2 = "."+a2.split(".")[1]
        else:
            ta1 = int(a2)
            ta2 = 0
            
        if "." in b2:                    
            tb1 = int(b2.split(".")[0])
            tb2 = "."+b2.split(".")[1]
        else:
            tb1 = int(b2)
            tb2 = 0
        
        # print("fraction of a = ", float(str(ta2).replace("?","")))
        # print("fraction of b ", float(str(tb2).replace("?","")))


        if plane ==0 :
            tb1 = 0
        elif plane ==1 :
            tb1 = 49
        elif plane ==2 :
            ta1 = 49
        elif plane ==3 :
            ta1 = 0



        currPixels = []

        currPixels.append(int(ta1))
        currPixels.append(int(tb1))
        
      

        if (float(ta2) > 0.999 ):
            # ta1 = int(ta1)+1
            currPixels.append(int(ta1)+1)
            currPixels.append(int(tb1))
        elif(float(ta2) < .0001 ):
            currPixels.append(int(ta1)-1)
            currPixels.append(int(tb1))
                
        if (float(tb2) > 0.999 ):              
            currPixels.append(int(ta1))
            currPixels.append(int(tb1)+1)
              
        elif(float(tb2) < .0001 ):
            currPixels.append(int(ta1))
            currPixels.append(int(tb1)-1)



      

        return currPixels
    else:
        print("no sat image")
        p =[-1,-1]
        return p
         
            
       
def getVertexPixelValueZ3(xp,yp,zp,x,y,z):
    s3 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=True)
    set_option(precision=20)
    set_param('parallel.enable', True)
    s3.set("sat.local_search_threads", 28)
    s3.set("sat.threads", 28)
    s3.set("sat.threads", 28)
    s3.check()
    
    a,b = Reals('a b')
    cons1 = ( (((-68.39567*(x -xp) )/ (z -zp) )+ 24.5 ) == a )
    cons2 =  ( (((68.39567*(y -yp) )/ (z -zp) )+ 24.5 ) == b )
    
    # print(x,y,z)
    # print(xp, yp, zp)
    
    s3.add(simplify(And(cons1,cons2)))
    p = [0,0]
    if s3.check() == sat:
        m = s3.model()
        print("model from solver :",m)
        # a1 = str(eval("m[a].numerator_as_long()/m[a].denominator_as_long()"))
        # b1 = str(eval("m[b].numerator_as_long()/m[b].denominator_as_long()"))
        # print(a1,b1) 
        
        a2 = str(eval("m[a]")).replace("?", "")
        b2 = str(eval("m[b]")).replace("?", "")
        # print("a2 = ", a2)
        # print("b2 = ", b2)
        
        # a2 = a2.split(".")[0]
        # b2 = b2.split(".")[0]
        # print("a2 = ", a2)
        # print("b2 = ", b2)
        
        if ("-" in a2 or "-" in b2 ):
            return [-1,-1]
        
        
        if "." in a2:
            ta1 = int(a2.split(".")[0])
            ta2 = "."+a2.split(".")[1]
        else:
            ta1 = int(a2)
            ta2 = 0
            
        if "." in b2:                    
            tb1 = int(b2.split(".")[0])
            tb2 = "."+b2.split(".")[1]
        else:
            tb1 = int(b2)
            tb2 = 0
        
        # print("fraction of a = ", float(str(ta2).replace("?","")))
        # print("fraction of b ", float(str(tb2).replace("?","")))
       
        global stopRoundingFlag
        # print("stopRounding Flag = ", stopRoundingFlag)
        # print("ta2 = ", ta2)
        # print("tb2 = ", tb2)
        # print("ta1 = ", ta1)
        # print("tb1 = ", tb1)



        currPixels = []

        currPixels.append(int(ta1))
        currPixels.append(int(tb1))
        
        # if (float(ta2) > 0.999):
        #     # ta1 = int(ta1)+1
        #     currPixels.append(int(ta1)+1)
        #     currPixels.append(int(tb1))
        # elif(float(ta2) < .0001):
        #     currPixels.append(int(ta1)-1)
        #     currPixels.append(int(tb1))
                
        # if (float(tb2) > 0.999):              
        #     currPixels.append(int(ta1))
        #     currPixels.append(int(tb1)+1)
              
        # elif(float(tb2) < .0001):
        #     currPixels.append(int(ta1))
        #     currPixels.append(int(tb1)-1)

      





        # if stopRoundingFlag == 0:
        #     if (float(ta2) > 0.999):
        #         ta1 = int(ta1)+1
            
        #     else:
        #         ta1 = int(ta1)
            
        #     if (float(tb2) > 0.999):
        #         tb1 = int(tb1)+1
            
        #     else:
        #         tb1 = int(tb1)
                
        # if stopDownRoundingFlag == 1 and stopRoundingFlag ==1:
        #     if(float(ta2) < .0001):
        #         ta1 = int(ta1)-1
        #     if(float(tb2) < .0001):
        #         tb1 = int(tb1)-1
        
        # exit(0)
           
        
        # a= float(a)
        # b= float(b)
        
        # frac_a, whole_a = math.modf(a)
        # frac_b, whole_b = math.modf(b)
        # # print("fractionalPart ", frac, whole)
        
        # if(frac_a >= 0.99999999999999999):
        #     p[0] = math.floor(float(a))+1
        # else:
        #     p[0] = math.floor(float(a))
            
        # if(frac_a <= 0.000000000000000001):
        #     p[0] = math.floor(float(a))-1
        # else:
        #     p[0] = math.floor(float(a))
        
        # if(frac_b >= 0.9999999999999999999):
        #     p[1] = math.floor(float(b))+1
        # else:
        #     p[1] = math.floor(float(b))
        
        return currPixels
    else:
        print("no sat image")
        p = [-1,-1]
        # exit() 
        return p 
    

def getVertexPixelValuePython(xp,yp,zp,x,y,z):
    
   
    
    a= ((-68.39567*(x -xp) )/ (z -zp) )+ 24.5 
    b= ((68.39567*(y -yp) )/ (z -zp) )+ 24.5 
    
    p = [0,0]    
    p[0] = math.floor(float(a))
    p[1] = math.floor(float(b))
            
    return p

def generateTrianglePosInvRegCons(numberOfFullyInsideVertices,insideVertexDetailsToPPL,\
            numberOfIntersectingEdges,intersectingEdgeDataToPPL,m, numberOfInvRegions):
    # print("generating constraints for inv region")
    consToReturn = And(True)
    consList = []
    
    global currZ3VariableIndex
    u, v, w, g = Reals('u v w g') 
    p, q = Reals('p q') 
    
    # print("Processing fully inside vertices, # = ", numberOfFullyInsideVertices)
    for i in range(0, numberOfFullyInsideVertices):
        currentVertexIndex = insideVertexDetailsToPPL[i][0]
        
        x = vertices[currentVertexIndex*3+0]
        y = vertices[currentVertexIndex*3+1]
        z = vertices[currentVertexIndex*3+2]
        
        xpixel = insideVertexDetailsToPPL[i][1]
        ypixel = insideVertexDetailsToPPL[i][2]
        
        
        
        #original
        cons1 = "( (((-68.39567*(x -xp0) )/ (z -zp0) )+ 24.5 ) >= xpixel )"
        cons2 = "( (((-68.39567*(x -xp0) )/ (z -zp0) )+ 24.5 ) < xpixel+1 )"
        cons3 = "( (((68.39567*(y -yp0) )/ (z -zp0) )+ 24.5 ) >= ypixel )"
        cons4 = "( (((68.39567*(y -yp0) )/ (z -zp0) )+ 24.5 ) < ypixel+1 )"
        
        # cons1 = "( (((-68.39567*(x -xp0) )/ (z -zp0) ) ) >= xpixel -24.5 )"
        # cons2 = "( (((-68.39567*(x -xp0) )/ (z -zp0) ) ) < xpixel+1 -24.5 )"
      
        
        allCons = And(eval(cons1), eval(cons2), eval(cons3), eval(cons4))

        print(allCons)
        
        
        consList.append((str(eval(cons1)).replace("?","").replace("\n","")))
        consList.append((str(eval(cons2)).replace("?","").replace("\n","")))
        consList.append((str(eval(cons3)).replace("?","").replace("\n","")))
        consList.append((str(eval(cons4)).replace("?","").replace("\n","")))
        
        consToReturn = And(consToReturn, allCons)
    
    # print("consToReturn == ", consToReturn)
    # print("Intersecting edges started.")
    # print("Number of intersecting edges = ", numberOfIntersectingEdges)
    for i in range(0, numberOfIntersectingEdges):
    # for i in range(0,0):


        edgeId = intersectingEdgeDataToPPL[i][0]
        # if(edgeId ==5 or edgeId ==6 or edgeId ==7):
        #     continue
        # print("Current intersecting edge = ", i, edgeId)
        insideVertex = intersectingEdgeDataToPPL[i][1]
        outsideVertex = intersectingEdgeDataToPPL[i][2]

        planeId = eval(str(intersectingEdgeDataToPPL[i][3]))
        xpixel = eval(str(intersectingEdgeDataToPPL[i][4]))
        ypixel = eval(str(intersectingEdgeDataToPPL[i][5]))
        
        # print("xpixel = ",xpixel)
        # print("ypixel = ", ypixel)

        ix = intersectingEdgeDataToPPL[i][6]
        iy = intersectingEdgeDataToPPL[i][7]
        iz = intersectingEdgeDataToPPL[i][8]

        m2p = intersectingEdgeDataToPPL[i][9]
        m2q = intersectingEdgeDataToPPL[i][10]
        
        
        
        
        xv0 = vertices[insideVertex*3+0] -xp0
        yv0 = vertices[insideVertex*3+1] -yp0
        zv0 = vertices[insideVertex*3+2] -zp0
        # wv0 = -(vertices[insideVertex*3+2] - m[zp0])
        
        xv1 = vertices[outsideVertex*3+0] -xp0
        yv1 = vertices[outsideVertex*3+1] -yp0
        zv1 = vertices[outsideVertex*3+2] -zp0
        # wv1 =  - (vertices[outsideVertex*3+2] - m[zp0])
        
        x = m2p*xv0+m2q*xv1
        y = m2p*yv0+m2q*yv1
        z = m2p*zv0+m2q*zv1
        
        # x = (p0*xv0+(1-p0)*xv1)
        # y = (p0*yv0+(1-p0)*yv1)
        # z = (p0*zv0+(1-p0)*zv1)
        
        plane0_v0 = [0, 0.0, -1]
        plane0_v1 = [0.0, 0.0, -1]
        plane0_v2 = [0, 0, -1000]
        plane0_v3 = [0, 0, -1000]
        
        

        if planeId == 0:
            # print("top plane")
            plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
            plane0_v1 = [0.35820895522388063, 0.35820895522388063, -1]
            plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
            plane0_v3 = [358.20895522388063, 358.20895522388063, -1000]
            
         
            
            # ypixel = 0.0000585074626831908057050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778050778
            ypixel = 0
            cons1 =  "( (((-68.39567*(xl_list[{}]) )/ (zl_list[{}]) )+ 24.5 ) >= xpixel )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons2 = "( (((-68.39567*(xl_list[{}] ) )/ (zl_list[{}]) )+ 24.5 ) < xpixel+1 )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons3 = "( (((68.39567*(yl_list[{}]) )/ (zl_list[{}]) )+ 24.5 ) >= ypixel)".format(currZ3VariableIndex, currZ3VariableIndex)
            cons4 =  "( (((68.39567*(yl_list[{}]) )/ (zl_list[{}]) )+ 24.5 ) <= ypixel+0.0005 )".format(currZ3VariableIndex, currZ3VariableIndex)
            
            # cons3 = "( (((68.39567*(yl_list[{}]) )/ (zl_list[{}]) )+ 24.5 ) == ypixel)".format(currZ3VariableIndex, currZ3VariableIndex)
            # cons4 =  "( (((68.39567*(yl_list[{}]) )/ (zl_list[{}]) )+ 24.5 ) <= ypixel+0.0005 )".format(currZ3VariableIndex, currZ3VariableIndex)
            
          
            
        if planeId == 1:
            print("bottom plane")
            plane0_v0 = [-0.35820895522388063, -0.35820895522388063, -1]
            plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
            plane0_v2 = [-358.20895522388063, -358.20895522388063, -1000]
            plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
            
            # ypixel = 48.9999414925373168091942949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221949221
            ypixel = 49
            cons1 =  "( (((-68.39567*(xl_list[{}] ) )/ (zl_list[{}]) )+ 24.5 ) >= xpixel )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons2 = "( (((-68.39567*(xl_list[{}]) )/ (zl_list[{}]) )+ 24.5 ) < xpixel+1 )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons3 = "( (((68.39567*(yl_list[{}]) )/ (zl_list[{}]) )+ 24.5 ) <= ypixel )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons4 =  "( (((68.39567*(yl_list[{}]) )/ (zl_list[{}]) )+ 24.5 ) >= ypixel-0.0001 )".format(currZ3VariableIndex, currZ3VariableIndex)
            
            # cons3 = "( (((68.39567*(yl_list[{}]) )/ (zl_list[{}]) )+ 24.5 ) == ypixel )".format(currZ3VariableIndex, currZ3VariableIndex)
            
           
            
        if planeId == 2:
            # print("right plane")
            plane0_v0 = [0.35820895522388063, 0.35820895522388063, -1]
            plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
            plane0_v2 = [358.20895522388063, 358.20895522388063, -1000]
            plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
            
            xpixel = 49
            cons1 =  "( (((-68.39567*(xl_list[{}] ) )/ (zl_list[{}]) )+ 24.5 ) <= xpixel )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons2 = "( (((-68.39567*(xl_list[{}]  ) )/ (zl_list[{}]) )+ 24.5 ) >= xpixel-0.001 )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons3 = "( (((68.39567*(yl_list[{}] ) )/ (zl_list[{}]) )+ 24.5 ) >= ypixel )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons4 =  "( (((68.39567*(yl_list[{}] ) )/ (zl_list[{}]) )+ 24.5 ) < ypixel+1 )".format(currZ3VariableIndex, currZ3VariableIndex)
            
            
            
        if planeId == 3:
            print("left plane ")
            plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
            plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
            plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
            plane0_v3 = [-358.20895522388063, -358.20895522388063, -1000]
            # cons1 = "0-b == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
            
            # cons1 =  "( (((-68.39567*(xl - xp0) )/ (zl-zp0) )+ 24.5 ) >= xpixel )"
            # cons2 = "( (((-68.39567*(xl -xp0 ) )/ (zl -zp0) )+ 24.5 ) < xpixel+1 )"
            # cons3 = "( (((68.39567*(yl -yp0) )/ (zl-zp0) )+ 24.5 ) >= ypixel )"
            # cons4 =  "( (((68.39567*(yl -yp0) )/ (zl-zp0) )+ 24.5 ) < ypixel+1 )"
            
            # cons1 =  "( (((-68.39567*(xl ) )/ (zl) )+ 24.5 ) >= xpixel )"
            # cons2 = "( (((-68.39567*(xl  ) )/ (zl) )+ 24.5 ) < xpixel+1 )"
            xpixel = 0
            cons1 =  "( (((-68.39567*(xl_list[{}] ) )/ (zl_list[{}]) )+ 24.5 ) >= xpixel )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons2 = "( (((-68.39567*(xl_list[{}]  ) )/ (zl_list[{}]) )+ 24.5 ) < xpixel+0.001 )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons3 = "( (((68.39567*(yl_list[{}] ) )/ (zl_list[{}]) )+ 24.5 ) >= ypixel )".format(currZ3VariableIndex, currZ3VariableIndex)
            cons4 =  "( (((68.39567*(yl_list[{}] ) )/ (zl_list[{}]) )+ 24.5 ) < ypixel+1 )".format(currZ3VariableIndex, currZ3VariableIndex)
            
            
            
        if planeId == 5:
            plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
            plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
            plane0_v2 = [0.35820895522388063, -0.35820895522388063, -1]
            plane0_v3 = [0.35820895522388063, 0.35820895522388063, -1]
            cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
            cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"

        px0 = plane0_v0[0]
        py0 = plane0_v0[1]
        pz0 = plane0_v0[2]

        px1 = plane0_v1[0]
        py1 = plane0_v1[1]
        pz1 = plane0_v1[2]

        px2 = plane0_v2[0]
        py2 = plane0_v2[1]
        pz2 = plane0_v2[2]

        px3 = plane0_v3[0]
        py3 = plane0_v3[1]
        pz3 = plane0_v3[2]
        
           
        cons5 = "p_list[{}]+q_list[{}] == 1".format(currZ3VariableIndex, currZ3VariableIndex)
        cons6 = "(And(p_list[{}]>=0,q_list[{}]>=0))".format(currZ3VariableIndex, currZ3VariableIndex)
        cons7 = "(xl_list[{}] == (p_list[{}]*xv0+ q_list[{}]*xv1))".format(currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex)
        cons8 = "(yl_list[{}] == (p_list[{}]*yv0+ q_list[{}]*yv1))".format(currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex)
        cons9 = "(zl_list[{}] == (p_list[{}]*zv0+ q_list[{}]*zv1))".format(currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex)
    
        cons10 = "(u_list[{}]+v_list[{}]+w_list[{}]+g_list[{}] == 1)".format(currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex)
        cons11 = "(And(u_list[{}]>=0, v_list[{}]>=0, w_list[{}]>=0,g_list[{}]>=0))".format(currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex)
        
        
        cons12 = "(xk_list[{}] == (u_list[{}]*px0+v_list[{}]*px1+w_list[{}]*px2+g_list[{}]*px3))".format(currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex)
        cons13 = "(yk_list[{}] == (u_list[{}]*py0+v_list[{}]*py1+w_list[{}]*py2+g_list[{}]*py3))".format(currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex)
        cons14 = "(zk_list[{}] == (u_list[{}]*pz0+v_list[{}]*pz1+w_list[{}]*pz2+g_list[{}]*pz3))".format(currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex, currZ3VariableIndex)
        

        cons15 = "(xl_list[{}] == xk_list[{}])".format(currZ3VariableIndex, currZ3VariableIndex)
        cons16 = "(yl_list[{}] == yk_list[{}])".format(currZ3VariableIndex, currZ3VariableIndex)
        cons17 = "(zl_list[{}] == zk_list[{}])".format(currZ3VariableIndex, currZ3VariableIndex)
        
        
       
        
        allCons = And(eval(cons1), eval(cons2), eval(cons3), eval(cons4))
        
        # s101.add(allCons) 
        # print(s101.check())
        
        allCons = And(allCons, eval(cons5),eval(cons6),eval(cons7), eval(cons8),eval(cons9),eval(cons10),eval(cons11),eval(cons12),eval(cons13),eval(cons14),eval(cons15),eval(cons16),eval(cons17))
        
        # print("\n ", eval(cons1))
        # print(simplify(eval(cons1)))
        # exit()
        
      
        
        # s101.add(allCons) 
        # print(s101.check())
        
        # print(s101.model())
        
        # exit()
        
        existQuantifier = "p_list[{}],q_list[{}], zl_list[{}], zk_list[{}], yl_list[{}], yk_list[{}], xl_list[{}], xk_list[{}], u_list[{}],v_list[{}],w_list[{}],g_list[{}]".format(currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex,currZ3VariableIndex)
        
        allCons = Exists(eval(existQuantifier), allCons)
        # allCons = Exists(eval(existQuantifier), allCons)
        
        
        # print("\n befor elimination")
        # print(allCons)
        g  = Goal()
        g.add((allCons))
        # set_option(rational_to_decimal=False)
        
        t1 = Tactic('simplify')
        t2 = Tactic('qe')
        t3 = Tactic('qe2')
        t  = Then( t2, t1)
        
        
        # exit()
        
        consList.append((str(eval(cons1)).replace("?","").replace("\n","")))
        consList.append((str(eval(cons2)).replace("?","").replace("\n","")))
        consList.append((str(eval(cons3)).replace("?","").replace("\n","")))
        # consList.append((str(eval(cons4)).replace("?","").replace("\n","")))
        
        # consToReturn = And(consToReturn, allCons)
        consToReturn = And(consToReturn, t(g)[0][0])
        # print("consToReturn == ", consToReturn)    
        print("consToReturn == ", consToReturn)
        # exit()
        currZ3VariableIndex = currZ3VariableIndex+1
        
    return consToReturn, consList
        
    
    
    
def getVertexPixelValuePythonUsingMatric(xp,yp,zp,x,y,z):
    M = mProj
    print(M)
    print(x,y,z)
    out = [0,0,0,0]
    x=  x-xp
    y = y-yp
    z = z-zp
    
    out[0] = x * M[0][0] + y * M[1][0] + z * M[2][0] +  M[3][0]; 
    out[1] = x * M[0][1] + y * M[1][1] + z * M[2][1] +  M[3][1]; 
    out[2] = x * M[0][2] + y * M[1][2] + z * M[2][2] +  M[3][2]; 
    out[3] = x * M[0][3] + y * M[1][3] + z * M[2][3] +  M[3][3]; 
    
    t0 = out[0]/out[3]
    t1 = out[1]/out[3]
    t2 = out[2]/out[3]
    
   
   
    p = [0,0,0,0]
    p[0] = min(environment.imageWidth - 1, ((t0 + 1) * 0.5 * environment.imageWidth))
    p[1] = min(environment.imageHeight - 1,((1 - (t1 + 1) * 0.5) * environment.imageHeight))
    p[2] = t2
    # print(environment.imageWidth)
    # # print(out)
    # # print(out[0]/out[3])
    # print(p)
    # print("pixel values''''^")
    # print("\n\n")
    
    p[0] =math.floor(p[0])
    p[1] = math.floor(p[1])
    return p

def planeEdgeIntersectionUpdated(plane,insideVertex, outsideVertex,m, newCode = 0): 
    # print("____________________")
    # print(" Inside planeEdgeIntersection Updated function")
    # print(plane,insideVertex, outsideVertex)
    s1 = Solver()
    set_param('parallel.enable', True)
    # set_option(rational_to_decimal=False)
    # set_option(precision=20)
    set_param('parallel.enable', True)
    s1.set("sat.local_search_threads", 26)
    s1.set("sat.threads", 26)
    s1.set("timeout",10000)
    
 
    xv0 = 0
    yv0 = 0
    zv0 = 0
    
    
    if newCode == 0:    
        xv0 = vertices[insideVertex*3+0] - m[xp0]
        yv0 = vertices[insideVertex*3+1] - m[yp0]
        zv0 = vertices[insideVertex*3+2] - m[zp0]
        wv0 = -(vertices[insideVertex*3+2] - m[zp0])
        
        xv1 = vertices[outsideVertex*3+0] - m[xp0]
        yv1 = vertices[outsideVertex*3+1] - m[yp0]
        zv1 = vertices[outsideVertex*3+2] - m[zp0]
        wv1 =  - (vertices[outsideVertex*3+2] - m[zp0])
    
    else:
        
        xv0 = insideVertex[0] - m[xp0]
        yv0 = insideVertex[1] - m[yp0]
        zv0 = insideVertex[2] - m[zp0]
        
        xv1 = outsideVertex[0] - m[xp0]
        yv1 = outsideVertex[1] - m[yp0]
        zv1 = outsideVertex[2] - m[zp0]
                
    
    # print(xv0,yv0,zv0, xv1,yv1,zv1)
   
    
    p,q = Reals('p q')
    xl, yl, zl = Reals('xl yl zl')
    
    s1.add(p+q == 1)
    s1.add(And(p>=0,q>=0))
    
    s1.add(xl == (p*xv0+ q*xv1))
    s1.add(yl == (p*yv0+ q*yv1))
    s1.add(zl == (p*zv0+ q*zv1))
    
   
    
    xk, yk, zk = Reals('xk yk zk')
    u, v, w, g  = Reals('u v w g')
    s1.check()
    s1.add(u+v+w+g == 1)
    s1.add(And(u>=0, v>=0, w>=0,g>=0))
    # s1.check()
    
    x0 = 0
    y0 = 0
    z0 = 0

    x1 = 0
    y1 = 0
    z1 = 0 
    
    x2 = 0
    y2 = 0
    z2 = 0 
    
    x3 = 0
    y3 = 0
    z3 = 0 
    
    
    if(plane == 0):
        # print("checking intersection with the top plane")
        # dummyValue = -1000
        # plane0_v0 = [-canvasWidth*25.4/(2*focalLength),canvasHeight*25.4*1.34/(2*focalLength),focalLength]
        # plane0_v1 = [0,0,0]
        # plane0_v2 = [canvasWidth*25.4/(2*focalLength),canvasHeight*25.4*1.34/(2*focalLength),focalLength]    

        plane0_v0 = [-0.35820895522388063,0.35820895522388063,-1]
        plane0_v1 = [0.35820895522388063,0.35820895522388063,-1]
        plane0_v2 = [-358.20895522388063,358.20895522388063,-1000]    
        plane0_v3 = [358.20895522388063,358.20895522388063,-1000] 
        
        # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
    
        
        
        x0 = plane0_v0[0]
        y0 = plane0_v0[1]
        z0 = plane0_v0[2] 

        x1 = plane0_v1[0]
        y1 = plane0_v1[1]
        z1 = plane0_v1[2] 
        
        x2 = plane0_v2[0]
        y2 = plane0_v2[1]
        z2 = plane0_v2[2] 
        
        x3 = plane0_v3[0]
        y3 = plane0_v3[1]
        z3 = plane0_v3[2] 


        
    
    if(plane == 1):
        # print("checking intersection with the bottom plane")
           

        plane0_v0 = [-0.35820895522388063,-0.35820895522388063,-1]
        plane0_v1 = [0.35820895522388063,-0.35820895522388063,-1]
        plane0_v2 = [-358.20895522388063,-358.20895522388063,-1000]    
        plane0_v3 = [358.20895522388063,-358.20895522388063,-1000] 
        
        # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
    
        # xk, yk, zk = Reals('xk yk zk')
        # u, v, w, g  = Reals('u v w g')
        # s1.check()
        # s1.add(u+v+w+g == 1)
        # s1.add(And(u>=0, v>=0, w>=0,g>=0))
        # s1.check()
        
        x0 = plane0_v0[0]
        y0 = plane0_v0[1]
        z0 = plane0_v0[2] 

        x1 = plane0_v1[0]
        y1 = plane0_v1[1]
        z1 = plane0_v1[2] 
        
        x2 = plane0_v2[0]
        y2 = plane0_v2[1]
        z2 = plane0_v2[2] 
        
        x3 = plane0_v3[0]
        y3 = plane0_v3[1]
        z3 = plane0_v3[2] 


      
    
    if(plane == 2):
            # print("checking intersection with the right plane")
           

            plane0_v0 = [0.35820895522388063,0.35820895522388063,-1]
            plane0_v1 = [0.35820895522388063,-0.35820895522388063,-1]
            plane0_v2 = [358.20895522388063,358.20895522388063,-1000]    
            plane0_v3 = [358.20895522388063,-358.20895522388063,-1000] 
            
            # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
        
            # xk, yk, zk = Reals('xk yk zk')
            # u, v, w, g  = Reals('u v w g')
            # s1.check()
            # s1.add(u+v+w+g == 1)
            # s1.add(And(u>=0, v>=0, w>=0,g>=0))
            # s1.check()
            
            x0 = plane0_v0[0]
            y0 = plane0_v0[1]
            z0 = plane0_v0[2] 

            x1 = plane0_v1[0]
            y1 = plane0_v1[1]
            z1 = plane0_v1[2] 
            
            x2 = plane0_v2[0]
            y2 = plane0_v2[1]
            z2 = plane0_v2[2] 
            
            x3 = plane0_v3[0]
            y3 = plane0_v3[1]
            z3 = plane0_v3[2] 


           
    
    if(plane == 3):
        # print("checking intersection with the left plane")
           

        plane0_v0 = [-0.35820895522388063,0.35820895522388063,-1]
        plane0_v1 = [-0.35820895522388063,-0.35820895522388063,-1]
        plane0_v2 = [-358.20895522388063,358.20895522388063,-1000]    
        plane0_v3 = [-358.20895522388063,-358.20895522388063,-1000] 
        
        # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
    
        # xk, yk, zk = Reals('xk yk zk')
        # u, v, w, g  = Reals('u v w g')
        # s1.check()
        # s1.add(u+v+w+g == 1)
        # s1.add(And(u>=0, v>=0, w>=0,g>=0))
        # s1.check()
        
        x0 = plane0_v0[0]
        y0 = plane0_v0[1]
        z0 = plane0_v0[2] 

        x1 = plane0_v1[0]
        y1 = plane0_v1[1]
        z1 = plane0_v1[2] 
        
        x2 = plane0_v2[0]
        y2 = plane0_v2[1]
        z2 = plane0_v2[2] 
        
        x3 = plane0_v3[0]
        y3 = plane0_v3[1]
        z3 = plane0_v3[2] 


                         
    
    if(plane ==5):
        # print("checking intersection with the near plane")
           

        plane0_v0 = [-0.35820895522388063,0.35820895522388063,-1]
        plane0_v1 = [-0.35820895522388063,-0.35820895522388063,-1]
        plane0_v2 = [0.35820895522388063,-0.35820895522388063,-1]    
        plane0_v3 = [0.35820895522388063,0.35820895522388063,-1] 
        
        # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
    
        # xk, yk, zk = Reals('xk yk zk')
        # u, v, w, g  = Reals('u v w g')
        # s1.check()
        # s1.add(u+v+w+g == 1)
        # s1.add(And(u>=0, v>=0, w>=0,g>=0))
        # s1.check()
        
        x0 = plane0_v0[0]
        y0 = plane0_v0[1]
        z0 = plane0_v0[2] 

        x1 = plane0_v1[0]
        y1 = plane0_v1[1]
        z1 = plane0_v1[2] 
        
        x2 = plane0_v2[0]
        y2 = plane0_v2[1]
        z2 = plane0_v2[2] 
        
        x3 = plane0_v3[0]
        y3 = plane0_v3[1]
        z3 = plane0_v3[2] 


        
    
    
    
    s1.add(xk == (u*x0+v*x1+w*x2+g*x3))
    s1.add(yk == (u*y0+v*y1+w*y2+g*y3))
    s1.add(zk == (u*z0+v*z1+w*z2+g*z3))
    # s1.check()
    # if(s1.check() ==sat):
    #     m23 =s1.model()
    #     print(m23)

    s1.add(xl == xk)
    s1.add(yl == yk)
    s1.add(zl == zk)
    
    # for c in s1.assertions():
    #     print(c,"\n")
    
    result = s1.check()
    if(result ==sat):
        # print("intersecting with the plane using linear combination; and z3")
        m2 = s1.model()
        # print(m2)
        
        insideFraction = eval("m2[p].numerator_as_long()/m2[p].denominator_as_long() " )
        outsideFraction = eval("m2[q].numerator_as_long()/m2[q].denominator_as_long() " )
        
         
        intersectionPoint = [0,0,0,0]

        
        vertexPixelValue2 = getVertexPixelValueIntersectZ3(m2[p]*xv0+m2[q]*xv1,\
                                                           m2[p]*yv0+m2[q]*yv1,\
                                                           m2[p]*zv0+m2[q]*zv1, plane) 
        
        
        
        intersectionPoint = [0,0,0,0]
        if newCode == 0:
            xv0 = vertices[insideVertex*3+0] 
            yv0 = vertices[insideVertex*3+1] 
            zv0 = vertices[insideVertex*3+2] 
            wv0 = -(vertices[insideVertex*3+2] )
                            
            xv1 = vertices[outsideVertex*3+0] 
            yv1 = vertices[outsideVertex*3+1] 
            zv1 = vertices[outsideVertex*3+2]
            wv1 =  - (vertices[outsideVertex*3+2] )
            
            intersectionPoint[0] =  eval("(1- outsideFraction)*xv0+ outsideFraction*xv1")
            intersectionPoint[1] = eval("(1- outsideFraction)*yv0+ outsideFraction*yv1")
            intersectionPoint[2] = eval("(1- outsideFraction)*zv0+ outsideFraction*zv1")
            intersectionPoint[3] = eval("(1- outsideFraction)*wv0+ outsideFraction*wv1")
        else:
            
            xv0 = insideVertex[0] 
            yv0 = insideVertex[1] 
            zv0 = insideVertex[2] 
            
            xv1 = outsideVertex[0] 
            yv1 = outsideVertex[1] 
            zv1 = outsideVertex[2] 
        
            intersectionPoint[0] =  eval("(1- outsideFraction)*xv0+ outsideFraction*xv1")
            intersectionPoint[1] = eval("(1- outsideFraction)*yv0+ outsideFraction*yv1")
            intersectionPoint[2] = eval("(1- outsideFraction)*zv0+ outsideFraction*zv1")
            
        
      
        
       
        
      
        return 1 , vertexPixelValue2, intersectionPoint,m2[p],m2[q]
    elif result == unsat:
        # 
        # print("Returning planeEdgeIntersection")
        return 0 , [-1,-1],[0,0,0],-1,-1  
    else:
       
        sleep(100)
      



    
def outValueToWorldCoordinates(out):
    x = out[0]
    y = out[1]
    z = out[2]
    
    a = np.array([
        [mProj[0][0], mProj[1][0], mProj[2][0]],
        [mProj[0][1], mProj[1][1], mProj[2][1]],
        [mProj[0][2], mProj[1][2], mProj[2][2]]
        ])
    
    b = np.array([x-mProj[3][0],y-mProj[3][1],z-mProj[3][2]])
    
    sol = np.linalg.solve(a,b)
    
    return sol
    
    

 
 
    
def computeOutcodeAtPos(i,outcodeP0, inx, iny,inz):
    
    outcode = 0   
    outx   = inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]
    outy   = inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    outz   = inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] 
    w      = inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3] 
    
    global outValues
    outValues[i*4+0] = outx
    outValues[i*4+1] = outy
    outValues[i*4+2] = outz
    outValues[i*4+3] = w
    
    outValueToReturn = [outx, outy, outz]
    # print(inx, iny, inz, outx, outy, outz,w)
    # print(mProj)
	
	# NFLRBT
    if( not(-w <= outx) ):
        # left
        # outcode=outcode ^ (1 << 3)
        outcodeP0[i*6+3] =1
        # outcode[3] = 1;
    if(not(outx <=w) ):
		# //right
        # outcode=outcode ^ (1 << 2)
        outcodeP0[i*6+2] =1
		# outcode[2] = 1;
    
    if( not(-w <= outy) ):
		# //bottom
        # outcode[1] = 1;
        # outcode=outcode ^ (1 << 1)
        outcodeP0[i*6+1] =1
    if(not(outy <=w) ):
		# //top
        # outcode=outcode ^ (1 << 0)
        outcodeP0[i*6+0] =1
		# outcode[0] = 1;
    
    if( not(-w <= outz) ):
		# //near
        # print("vertex outside near plane")
        # sleep(1)
        # outcode[5] = 1;
        # outcode=outcode ^ (1 << 5)
        outcodeP0[i*6+5] =1
    if(not(outz <=w) ):
		# //far
        # outcode[4] = 1;
        # outcode=outcode ^ (1 << 4)
        outcodeP0[i*6+4] =1
        
    return outValueToReturn, w

def symComputeOutcodePlane(plane, inx, iny,inz):
    
    #print(plane, inx, iny,inz)
    # print(mProj)
    inchToMm = 25.4 
    
    
    set_option(rational_to_decimal=True)
    
    if(plane == 0):
        return If(inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] >\
                  inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3],\
                    1,0)
    
    if(plane == 1):
        return If( (- ( inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3]) >\
                    (inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1])),\
                    1,0)
        # inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    
    if(plane == 2):
        return If((inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0] >\
                   inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3]),\
                       1,0)
    if(plane == 3):
        return If( (- ( inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3])>\
                    inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]),\
                        1,0)
    
    if(plane == 4):
        return If((inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] >\
                 inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3]),\
                     1,0)
    
    if(plane == 5):
        return If( (- ( inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3]) >\
                 inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2]),\
                     1,0)
    
    
def symComputeOutcodePlane2(plane, inx, iny,inz):
    
    #print(plane, inx, iny,inz)
    # print(mProj)
    inchToMm = 25.4
    yScale = Real('yScale')
    top = Real('top')
    right = Real('right')
    bottom = Real('bottom')
    left = Real('left')

    m00, m11, m22, m32 = Reals('m00 m11 m22 m32') 
    
    mProj = [
        [2 * n / (r - l), 0, 0, 0],
        [0,2 * n / (t - b),0,0],
        [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1 ],
        [0,0,-2 * f * n / (f - n),0]
    ]
    
    s10 = Solver()
    cons1 = (top == ( ( (canvasHeight * inchToMm / 2) / focalLength) ) * (yScale) )   # 0.735 * 25.4 / 2 / 35
    cons2 = (right == ((canvasWidth * inchToMm / 2) / focalLength))
    cons3 = (bottom == -top)
    cons4 = (left == -right)
    cons5 = (m00 == 2 / (right - left))
    cons6 = (yScale == canvasWidth/canvasHeight)
    cons7 = (m11 == 2 / (top - bottom))
    cons8 = (m22 == -(f + n) / (f - n))
    cons9 = (m32 == -2 * f * n / (f - n))


    essentialCons = And(cons1, cons2, cons3, cons4, cons5, cons6, cons7, cons8, cons9)


    s10.add(essentialCons)
    s10.check()
    m10 = s10.model()
    
    
    
    set_option(rational_to_decimal=True)
    
    if(plane == 0):
        return If(inx * mProj[0][1] + iny * m10[m11] + inz * mProj[2][1] +  mProj[3][1] >\
                  inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3],\
                    1,0)
    
    if(plane == 1):
        return If( (- ( inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3]) >\
                    (inx * mProj[0][1] + iny * m10[m11] + inz * mProj[2][1] +  mProj[3][1])),\
                    1,0)
        # inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    
    if(plane == 2):
        return If((inx * m10[m00] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0] >\
                   inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3]),\
                       1,0)
    if(plane == 3):
        return If( (- ( inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3])>\
                    inx * m10[m00] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]),\
                        1,0)
    
    if(plane == 4):
        return If((inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] >\
                 inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3]),\
                     1,0)
    
    if(plane == 5):
        return If( (- ( inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3]) >\
                 inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2]),\
                     1,0)
   

def findEdges(currVetex):
    flag = 0
    edge1 = 0
    edge2 = 0
    
    for currEdge in edges:
        if(currVetex == currEdge[0] or currVetex == currEdge[1]):
            if(flag == 0):
                if (currVetex != currEdge[0]):
                    edge1 = currEdge[0]
                else:
                    edge1 = currEdge[1]
                flag = 1
            else:
                if (currVetex != currEdge[0]):
                    edge2 = currEdge[0]
                else:
                    edge2 = currEdge[1]
    return edge1, edge2
    
    

def removeEdges(outsideVertex):
    #TODO: we can rewrite this function, but for now let it be, we change it later
    numberOfOccur = 0
    
    for currEdge in edges:
        if(currEdge[0] == outsideVertex or currEdge[1] == outsideVertex):
            numberOfOccur+=1
    
    for i in range(0,numberOfOccur):
        for currEdge in edges:
            if(currEdge[0] == outsideVertex or currEdge[1] == outsideVertex):
                edges.remove(currEdge)
                break
         
def cpp_vertexPlaneIntersectionPoint(insideVertex, outsideVertex, insidevertexW, outsideVertexW, intersectionPlane):

    x1 = insideVertex[0]
    y1 = insideVertex[1]
    z1 = insideVertex[2]
    w1 = insidevertexW

    x2 = outsideVertex[0]
    y2 = outsideVertex[1]
    z2 = outsideVertex[2]
    w2 = outsideVertexW

    #3 => w=-x , 2 => w=x, 1 => w=-y, 0=> w=y, 5 ==>w=-z near, 4==>w=far
    if(intersectionPlane == 3):
        t1= (-w1-x1)/(w2-w1+x2-x1)
    elif(intersectionPlane == 2):
        t1= (x1-w1)/(w2-w1-x2+x1)
    elif(intersectionPlane == 1):
        t1 = (-w1-y1)/(w2-w1+y2-y1)
    elif(intersectionPlane == 0):
        t1 = (-w1+y1)/(w2-w1-y2+y1)
    elif(intersectionPlane == 5):
        t1 = (-1-w1)/(w2-w1)
    elif(intersectionPlane == 4):   
        t1 = (-1000-w1)/(w2-w1)
        
        
    intersectionPoint = [x1+t1*(x2-x1), y1+ t1*(y2-y1), z1+t1*(z2-z1)]
    intersectionPointW = w1+t1*(w2-w1)
    # print("intersectionPoint = ",intersectionPoint)
    

    return t1, intersectionPoint, intersectionPointW 

def clipCoordinateToOutcode(out,  w):
    
    outcode = [0]*6
    if( not (-w <= out[0]) ):
        outcode[3] = 1
    elif(not (out[0] <=w) ):
        outcode[2] = 1
        
    if( not (-w <= out[1]) ):
        outcode[1] = 1
    elif(not (out[1] <=w) ):
        outcode[0] = 1
        
    if( not(-w <= out[2]) ):
        outcode[5] = 1
    elif(not (out[2] <=w) ):
        outcode[4] = 1
    
    return outcode        
  


def pixelValue(point, w):
    t0 = point[0]/w
    t1 = point[1]/w
    t2 = point[2]/w
    
    # print(((t0 + 1) * 0.5 * imageWidth),((1 - (t1 + 1) * 0.5) * imageHeight), t2)
    originalPixel = [int((t0 + 1) * 0.5 * imageWidth),int((1 - (t1 + 1) * 0.5) * imageHeight), t2]
    # print("pixel value before min = ",originalPixel  )
    
    raster0 = min(imageWidth - 1, int((t0 + 1) * 0.5 * imageWidth))
    raster1 = min(imageHeight - 1, int((1 - (t1 + 1) * 0.5) * imageHeight))
    raster2 = t2
    
    
    a2 = str((t0 + 1) * 0.5 * imageWidth)
    b2 =str((1 - (t1 + 1) * 0.5) * imageHeight)
     
       
        
    if "." in a2:
        ta1 = int(a2.split(".")[0])
        ta2 = "."+a2.split(".")[1]
    else:
        ta1 = int(a2)
        ta2 = 0
        
    if "." in b2:                    
        tb1 = int(b2.split(".")[0])
        tb2 = "."+b2.split(".")[1]
    else:
        tb1 = int(b2)
        tb2 = 0


        
     
    # # print("ta2 = ", ta2)
    # # print("tb2 = ", tb2)
    # # print("ta1 = ", ta1)
    # # print("tb1 = ", tb1)
    currPixels = []

    currPixels.append(int(ta1))
    currPixels.append(int(tb1))
    
    if (float(ta2) > 0.999):
        # ta1 = int(ta1)+1
        currPixels.append(int(ta1)+1)
        currPixels.append(int(tb1))
    elif(float(ta2) < .0001 ):
        currPixels.append(int(ta1)-1)
        currPixels.append(int(tb1))
            
    if (float(tb2) > 0.999 ):              
        currPixels.append(int(ta1))
        currPixels.append(int(tb1)+1)
            
    elif(float(tb2) < .0001 ):
        currPixels.append(int(ta1))
        currPixels.append(int(tb1)-1)

    

    


   
    
    return currPixels,originalPixel 

   


def prepareFinalIntervalImage(globalIntervalImage):
    finalGlobalIntervalImage.clear()
    
    for currPixel, currData in globalIntervalImage.items():
        tempData = []
        for i in range(1, len(currData)):
            if currData[i][6] <= currData[0][0]:
                tempData.append(currData[i])
        
        rmin = min([x[0] for x in tempData])
        rmax = max([x[1] for x in tempData])
        
        gmin = min([x[2] for x in tempData])
        gmax = max([x[3] for x in tempData])
        
        bmin = min([x[4] for x in tempData])
        bmax = max([x[5] for x in tempData])
        
        finalGlobalIntervalImage[currPixel] = [rmin, rmax, gmin, gmax, bmin, bmax]
        
    
    

def computeTriangleInvariantRegions2(currTriangle,currGroupName, currGroupRegionCons):
    
    s2 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=True)
    # set_option(precision=1000)
    set_param('parallel.enable', True)
    s2.set("sat.local_search_threads", 28)
    s2.set("sat.threads", 28)
    # s2.set("timeout",2000)    
    set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)

    s2.add(simplify(currGroupRegionCons))
    
    
    vertex0 = nvertices[currTriangle*3+0]
    vertex1 = nvertices[currTriangle*3+1]
    vertex2 = nvertices[currTriangle*3+2]
    currTriangleVertices = [vertex0, vertex1,vertex2]
    
    
   


    v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
    v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
    v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]


    if Counter(v0Vertex) == Counter(v1Vertex) or Counter(v0Vertex) == Counter(v2Vertex) or Counter(v1Vertex) == Counter(v2Vertex):
       
        return 0
    
    
    numberOfInvRegions = 0  
    
    
    if(vertices[currTriangleVertices[0]*3+2] > posZp100+1 and vertices[currTriangleVertices[1]*3+2] > posZp100+1 and vertices[currTriangleVertices[2]*3+2] > posZp100+1 ):
        return 0

   


    vertex_plane_cons = [
        [And(True),And(True),And(True),And(True),And(True),And(True)],
        [And(True),And(True),And(True),And(True),And(True),And(True)],
        [And(True),And(True),And(True),And(True),And(True),And(True)],                 
                         ]
    
    # print(vertex_plane_cons)
    
   
    for l in range(0,3):
        for m in range(0,6):
            # print(l,m)
            vertex_plane_cons[l][m] = symComputeOutcodePlane(m, vertices[currTriangleVertices[l]*3+0] -xp0,
                                                             vertices[currTriangleVertices[l]*3+1] -yp0,
                                                             vertices[currTriangleVertices[l]*3+2]-zp0)
            # print(vertex_plane_cons[0][1])

    vertex0Inside_cons = And (vertex_plane_cons[0][0] == 0, vertex_plane_cons[0][1] == 0,vertex_plane_cons[0][2] == 0,
                            vertex_plane_cons[0][3] == 0,vertex_plane_cons[0][5] == 0,vertex_plane_cons[0][4] == 0)
    
    vertex1Inside_cons = And (vertex_plane_cons[1][0] == 0, vertex_plane_cons[1][1] == 0,vertex_plane_cons[1][2] == 0,
                            vertex_plane_cons[1][3] == 0,vertex_plane_cons[1][5] == 0,vertex_plane_cons[1][4] == 0)
    
    vertex2Inside_cons = And (vertex_plane_cons[2][0] == 0, vertex_plane_cons[2][1] == 0,vertex_plane_cons[2][2] == 0,
                            vertex_plane_cons[2][3] == 0,vertex_plane_cons[2][5] == 0, vertex_plane_cons[2][4] == 0)
    
    anyOneVertexFullyInside = Or(vertex0Inside_cons,vertex1Inside_cons,vertex2Inside_cons)
    
    
    
    if tedges[(currTriangle) *6+0] == vertex0 :
            edge0_v0 = 0
    elif tedges[(currTriangle) *6+0] == vertex1 :
        edge0_v0 = 1
    else:
        edge0_v0 = 2
    
    if tedges[ (currTriangle) *6+1] == vertex0 :
        edge0_v1 = 0
    elif tedges[(currTriangle) *6+1] == vertex1 :
        edge0_v1 = 1
    else:
        edge0_v1 = 2
        
        
    if tedges[ (currTriangle) *6+2] == vertex0 :
        edge1_v0 = 0
    elif tedges[ (currTriangle) *6+2] == vertex1 :
        edge1_v0 = 1
    else:
        edge1_v0 = 2
        
    if tedges[ (currTriangle) *6+3] == vertex0 :
        edge1_v1 = 0
    elif tedges[ (currTriangle) *6+3] == vertex1 :
        edge1_v1 = 1
    else:
        edge1_v1 = 2
        
    
    if tedges[ (currTriangle) *6+4] == vertex0 :
        edge2_v0 = 0
    elif tedges[ (currTriangle) *6+4] == vertex1 :
        edge2_v0 = 1
    else:
        edge2_v0 = 2 
        
    if tedges[(currTriangle) *6+5] == vertex0 :
        edge2_v1 = 0
    elif tedges[ (currTriangle) *6+5] == vertex1 :
        edge2_v1 = 1
    else:
        edge2_v1 = 2  


    edgeVertexIndices = [edge0_v0, edge0_v1,edge1_v0, edge1_v1, edge2_v0, edge2_v1 ]
    
    
   
    
    
    #
    
    edge0_fullyOutside_cons = Or (
            And( vertex_plane_cons[edge0_v0][0] ==1, vertex_plane_cons[edge0_v1][0] ==1),
            And( vertex_plane_cons[edge0_v0][1] ==1, vertex_plane_cons[edge0_v1][1] ==1),
            And( vertex_plane_cons[edge0_v0][2] ==1, vertex_plane_cons[edge0_v1][2] ==1),
            And( vertex_plane_cons[edge0_v0][3] ==1, vertex_plane_cons[edge0_v1][3] ==1), 
            And( vertex_plane_cons[edge0_v0][4] ==1, vertex_plane_cons[edge0_v1][4] ==1),
            And( vertex_plane_cons[edge0_v0][5] ==1, vertex_plane_cons[edge0_v1][5] ==1)
            
            )
    
    edge1_fullyOutside_cons = Or (
                And( vertex_plane_cons[edge1_v0][0] ==1, vertex_plane_cons[edge1_v1][0] ==1),
                And( vertex_plane_cons[edge1_v0][1] ==1, vertex_plane_cons[edge1_v1][1] ==1),
                And( vertex_plane_cons[edge1_v0][2] ==1, vertex_plane_cons[edge1_v1][2] ==1),
                And( vertex_plane_cons[edge1_v0][3] ==1, vertex_plane_cons[edge1_v1][3] ==1), 
                And( vertex_plane_cons[edge1_v0][4] ==1, vertex_plane_cons[edge1_v1][4] ==1),
                And( vertex_plane_cons[edge1_v0][5] ==1, vertex_plane_cons[edge1_v1][5] ==1)
    )
    
    edge2_fullyOutside_cons = Or (
                And( vertex_plane_cons[edge2_v0][0] ==1, vertex_plane_cons[edge2_v1][0] ==1),
                And( vertex_plane_cons[edge2_v0][1] ==1, vertex_plane_cons[edge2_v1][1] ==1),
                And( vertex_plane_cons[edge2_v0][2] ==1, vertex_plane_cons[edge2_v1][2] ==1),
                And( vertex_plane_cons[edge2_v0][3] ==1, vertex_plane_cons[edge2_v1][3] ==1), 
                And( vertex_plane_cons[edge2_v0][4] ==1, vertex_plane_cons[edge2_v1][4] ==1),
                And( vertex_plane_cons[edge2_v0][5] ==1, vertex_plane_cons[edge2_v1][5] ==1),
                
                )

    atleastOneEdgeIntersect_cons = Not(And(edge0_fullyOutside_cons, edge1_fullyOutside_cons, edge2_fullyOutside_cons))
    
    triangleInsideOutsideCons = Or(anyOneVertexFullyInside, atleastOneEdgeIntersect_cons)
    
    # s2.push()
    s2.add(simplify(triangleInsideOutsideCons))
    
    # print(s2.check())
    # print(s2.model())
    
    s2.push()
    
    allCamPoses = dict()
   
    
    
    
    currTriangleInvPositions = [0]*10000*3
    currTriangleInvDepths = [0]*10000*2
    
    
    

    globalCurrentImage.clear()
    globalInsideVertexDataToPPL.clear()
    globalIntersectingVertexDataToPPL.clear()
    
    dataToComputeIntervalImage.clear()
    currTriangleIntervalImage.clear()
    
  

   
    
    currVarsIndex = 0
   
    while(s2.check() ==sat):
    # if(s2.check() ==sat):
        
      
        
        
        currListOfConsToAdd = []
        currListOfVertColours = dict()
        
        currImage =[]
        currImageColours = dict()
        currImageColours.clear()
        
        currImageColours[0] = [environment.vertColours[currTriangleVertices[0]*3+0],
                               environment.vertColours[currTriangleVertices[0]*3+1],
                               environment.vertColours[currTriangleVertices[0]*3+2]] 
        
        currImageColours[1] = [environment.vertColours[currTriangleVertices[1]*3+0],
                               environment.vertColours[currTriangleVertices[1]*3+1],
                               environment.vertColours[currTriangleVertices[1]*3+2]] 
        
        currImageColours[2] = [environment.vertColours[currTriangleVertices[2]*3+0],
                               environment.vertColours[currTriangleVertices[2]*3+1],
                               environment.vertColours[currTriangleVertices[2]*3+2]] 
        
        
        
        newvertices = [0]*numOfVertices*3*5
        newVerticesNumber = 0
        pixelValueComputed = [0]*numOfVertices*5
        pixelValues = [0]*numOfVertices*2*5
        
        edgesInSmallPyramid = [0]*numOftedges
        
        insideVertexDetailsToPPL = [] #store vertex index number,xpixel,ypixel
        numberOfFullyInsideVertices = 0
        numberOfIntersectingEdges = 0
        intersectingEdgeDataToPPL = []
        
        fullyInsideVerticesNumber = []
        intersectingVerticesNumber = []
        
        intersectingData = dict()
        intersectingData.clear()

        globalCurrentImage.clear()
        globalInsideVertexDataToPPL.clear()
        globalIntersectingVertexDataToPPL.clear()
        
        edges.clear()
        
        tr_num_of_vertices = 3
        tr_curr_num_of_vertex = 3
        tr_vertex_coordinates = np.zeros((100,3))
        tr_vertex_ws = [0]*100
        tr_vertex_outcodes = np.zeros((100,6))
        tr_vertices_set = []
        
        
        # print(datetime.now())
        # print(s2.check())
        # set_option(rational_to_decimal=False)
        m = s2.model()
        
        # sleep(1)
        
        # print("is_rational_value(m[xp0]):", is_rational_value(m[xp0]))
        
        # r0 = m[xp0].approx(20)
        # r1 = m[yp0].approx(20)
        # r2 = m[zp0].approx(20)
        
        posXp = (eval("m[xp0].numerator_as_long()/m[xp0].denominator_as_long()"))
        posYp = (eval("m[yp0].numerator_as_long()/m[yp0].denominator_as_long()"))
        posZp = (eval("m[zp0].numerator_as_long()/m[zp0].denominator_as_long()"))

        posXpF = float(posXp)
        posYpF = float(posYp)
        posZpF = float(posZp)

       
        # posXp = float(r0.numerator_as_long())/float(r0.denominator_as_long())
        # posYp = float(r1.numerator_as_long())/float(r1.denominator_as_long())
        # posZp = float(r2.numerator_as_long())/float(r2.denominator_as_long())
    
        notTheCurrentPosCons1 = Or(xp0!= m[xp0], yp0!=m[yp0],zp0!= m[zp0])
        currentPlane = 0 #top plane
        # sleep(0.5)
        
        
        # exit()
       
        # outcodeP0 = [0]*numOfVertices*6
        outcodeP0 = [0]*30*6
            
        
        
        outValue0, outW0 = computeOutcodeAtPos(0,outcodeP0, 
                            vertices[currTriangleVertices[0]*3+0]-posXp ,
                            vertices[currTriangleVertices[0]*3+1]-posYp,
                            vertices[currTriangleVertices[0]*3+2]-posZp )
        outValue1, outW1 = computeOutcodeAtPos(1,outcodeP0, 
                            vertices[currTriangleVertices[1]*3+0]-posXp ,
                            vertices[currTriangleVertices[1]*3+1]-posYp,
                            
                            vertices[currTriangleVertices[1]*3+2]-posZp )
        
        outValue2, outW2 = computeOutcodeAtPos(2,outcodeP0, 
                            vertices[currTriangleVertices[2]*3+0]-posXp ,
                            vertices[currTriangleVertices[2]*3+1]-posYp,
                            vertices[currTriangleVertices[2]*3+2]-posZp )
        
        
       
        
        bit0 = outcodeP0[0] & outcodeP0[6] & outcodeP0[12]
        bit1 = outcodeP0[1] & outcodeP0[7] & outcodeP0[13]
        bit2 = outcodeP0[2] & outcodeP0[8] & outcodeP0[14]
        bit3 = outcodeP0[3] & outcodeP0[9] & outcodeP0[15]
        bit4 = outcodeP0[4] & outcodeP0[10] & outcodeP0[16]
        bit5 = outcodeP0[5] & outcodeP0[11] & outcodeP0[17]
        
        
        
        tr_vertex_outcodes[0] = outcodeP0[0:6]
        tr_vertex_outcodes[1] = outcodeP0[6:12]
        tr_vertex_outcodes[2] = outcodeP0[12:18]

        tr_vertex_coordinates[0] = outValue0
        tr_vertex_coordinates[1] = outValue1
        tr_vertex_coordinates[2] = outValue2

        tr_vertex_ws[0]=outW0
        tr_vertex_ws[1]=outW1
        tr_vertex_ws[2]=outW2


        tr_vertices_set.append(0)
        tr_vertices_set.append(1)
        tr_vertices_set.append(2)

        # edges.append([0,1])
        # edges.append([0,2])
        # edges.append([1,2])
        
        edges.append([edge0_v0,edge0_v1])
        edges.append([edge1_v0,edge1_v1])
        edges.append([edge2_v0,edge2_v1])
            
                
        if(not any(outcodeP0)):
            
            for currVert in range(0,3):        
                
                # pixelValueComputed[edge_v0] = 1
                
                x = vertices[currTriangleVertices[currVert]*3+0]
                y = vertices[currTriangleVertices[currVert]*3+1]
                z = vertices[currTriangleVertices[currVert]*3+2]
                # vertexPixelValue = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                # print("vertexPixelValue ==> ",x,y,z, vertexPixelValue)
                
         
                
                currVertColour = environment.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
                currListOfVertColours[currVert] = currVertColour
                
         



                if currVert == 0:
                    vertexPixelValue, temp = pixelValue(outValue0,outW0)
                elif currVert == 1:
                    vertexPixelValue, temp = pixelValue(outValue1,outW1)
                else:
                    vertexPixelValue, temp = pixelValue(outValue2,outW2)
                
               
                

                fflag = 0
                currVertexPixelData = [] 
                for vPixels in range(0,len(vertexPixelValue),2):
                    if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
                        currVertColour = environment.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
                        tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert,currVertColour]
                        insideVertexDetailsToPPL.append(tempPixelData)
                        currVertexPixelData.append(tempPixelData)                            
                        if fflag == 0:
                            numberOfFullyInsideVertices+=1
                            fflag = 1
                        currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
                        
                        # print("Adding data to PPL insideVertexDetailsToPPL")
                if fflag == 1:
                    globalInsideVertexDataToPPL.append(currVertexPixelData)
                    fullyInsideVerticesNumber.append(currVert)
        
        elif(bit0 or bit1 or bit2 or bit3 or bit4 or bit5):            
            
            return 0
        else:
            
                  
            
            #find data of the fully inside vertices
            vert0_pos = outcodeP0[0] | outcodeP0[1] | outcodeP0[2] | outcodeP0[3] | outcodeP0[4] | outcodeP0[5]
            vert1_pos = outcodeP0[6] | outcodeP0[7] | outcodeP0[8] | outcodeP0[9] | outcodeP0[10] | outcodeP0[11]
            vert2_pos = outcodeP0[12] | outcodeP0[13] | outcodeP0[14] | outcodeP0[15] | outcodeP0[16] | outcodeP0[17]
            
            
            if(not vert0_pos):
                currVert =0
               
                
                
                
                vertexPixelValue, temp = pixelValue(outValue0,outW0)  
                

                fflag = 0
                currVertexPixelData = [] 
                for vPixels in range(0,len(vertexPixelValue),2):
                    if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
                        currVertColour = environment.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
                        tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert,currVertColour]
                        insideVertexDetailsToPPL.append(tempPixelData)
                        currVertexPixelData.append(tempPixelData)                            
                        if fflag == 0:
                            numberOfFullyInsideVertices+=1
                            fflag = 1
                        currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
                        # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
                        # print("Adding data to PPL insideVertexDetailsToPPL")
                if fflag == 1:
                    globalInsideVertexDataToPPL.append(currVertexPixelData)
                    fullyInsideVerticesNumber.append(currVert)
                
            
            if(not vert1_pos):
                currVert =1
                
                
               
                
                vertexPixelValue, temp = pixelValue(outValue1,outW1)   
                fflag = 0
                currVertexPixelData = [] 
                for vPixels in range(0,len(vertexPixelValue),2):
                    if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
                        currVertColour = environment.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
                        tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert, currVertColour]
                        insideVertexDetailsToPPL.append(tempPixelData)
                        currVertexPixelData.append(tempPixelData)                            
                        if fflag == 0:
                            numberOfFullyInsideVertices+=1
                            fflag = 1
                        currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
                        # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
                        # print("Adding data to PPL insideVertexDetailsToPPL")
                if fflag == 1:
                    globalInsideVertexDataToPPL.append(currVertexPixelData)
                    fullyInsideVerticesNumber.append(currVert)
            
            if(not vert2_pos):
                currVert =2
              
                
                vertexPixelValue, temp = pixelValue(outValue2,outW2)  
                fflag = 0
                currVertexPixelData = [] 
                for vPixels in range(0,len(vertexPixelValue),2):
                    if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
                        currVertColour = environment.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
                        tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert, currVertColour]
                        insideVertexDetailsToPPL.append(tempPixelData)
                        currVertexPixelData.append(tempPixelData)                            
                        if fflag == 0:
                            numberOfFullyInsideVertices+=1
                            fflag = 1
                        currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
                        # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
                        # print("Adding data to PPL insideVertexDetailsToPPL")
                if fflag == 1:
                    globalInsideVertexDataToPPL.append(currVertexPixelData)
                    fullyInsideVerticesNumber.append(currVert)
        
            num_of_planes = 6
            
            
            # print("tr_vertex_outcodes = ", tr_vertex_outcodes)
            
            # for currPlane in range(0, num_of_planes):
            for currPlane in [5,0,1,2,3]:
                
                # print("Current plane = ", currPlane)
                tr_outside_vertex_set = []
                num_of_outside_vertices = 0
                
                for currVert in tr_vertices_set:
                    if(tr_vertex_outcodes[currVert][currPlane] == 1):
                       
                        num_of_outside_vertices +=1
                        tr_outside_vertex_set.append(currVert)
                
               
                
                while(num_of_outside_vertices > 2):
                    
                    for currOutsideVert in tr_outside_vertex_set:               
                        edge1, edge2 = findEdges(currOutsideVert)
                       
                        if(tr_outside_vertex_set.count(edge1) and tr_outside_vertex_set.count(edge2)):
                           
                            removeEdges(currOutsideVert)
                            tr_vertices_set.remove(currOutsideVert)
                            tr_outside_vertex_set.remove(currOutsideVert)
                            tr_num_of_vertices = tr_num_of_vertices-1
                            edges.append([edge1,edge2])
                            break
                    num_of_outside_vertices -=1
                
               
                if num_of_outside_vertices == 1:
                    
                    
                    outsideVertex = tr_outside_vertex_set[0]
                   
                    edge1,edge2 = findEdges(outsideVertex)
                    
                    
                   

                    insideVertex1 = tr_vertex_coordinates[edge1]
                    insideVertex1W = tr_vertex_ws[edge1]
                    insideVertex2 = tr_vertex_coordinates[edge2]
                    insideVertex2W = tr_vertex_ws[edge2]
                    outsideVertex1 = tr_vertex_coordinates[outsideVertex]
                    outsideVertex1W =tr_vertex_ws[outsideVertex]
                    
                    
                    prop_t1, intersectionPoint1, intersectionPoint1W = cpp_vertexPlaneIntersectionPoint(insideVertex1, outsideVertex1, 
                                                                                        insideVertex1W, outsideVertex1W, currPlane)


                    

                    

                    prop_t2, intersectionPoint2, intersectionPoint2W = cpp_vertexPlaneIntersectionPoint(insideVertex2, outsideVertex1,
                                                                                        insideVertex2W,outsideVertex1W, currPlane)
                    
                    
                    
                    if currPlane == 5:
                        intersectionPoint1W = -intersectionPoint1W
                        intersectionPoint2W = -intersectionPoint2W
                
                    removeEdges(outsideVertex)
                    tr_vertices_set.remove(outsideVertex)
                    tr_num_of_vertices = tr_num_of_vertices-1
                    
                    
                    tr_vertex_coordinates[tr_curr_num_of_vertex] = intersectionPoint1
                    tr_vertex_ws[tr_curr_num_of_vertex] = intersectionPoint1W
                    tr_vertices_set.append(tr_curr_num_of_vertex)                             
                    outcode_t1 = clipCoordinateToOutcode(intersectionPoint1, intersectionPoint1W)
                    tr_vertex_outcodes[tr_curr_num_of_vertex] = outcode_t1
                    tr_num_of_vertices = tr_num_of_vertices+1  
                    
                    tempIntersectingData = [tr_curr_num_of_vertex, edge1, outsideVertex, currPlane]
                    intersectingData[tr_curr_num_of_vertex] = tempIntersectingData
                   
                    redIntvert = currImageColours[edge1][0] + prop_t1 * (currImageColours[edge1][0] - currImageColours[outsideVertex][0])
                    greenIntvert = currImageColours[edge1][1] + prop_t1 * (currImageColours[edge1][1] - currImageColours[outsideVertex][1])
                    blueIntvert = currImageColours[edge1][2] + prop_t1 * (currImageColours[edge1][2] - currImageColours[outsideVertex][2])
                                             
                    currImageColours[tr_curr_num_of_vertex] = [redIntvert, greenIntvert, blueIntvert]
                     
                    tr_curr_num_of_vertex += 1
                    
                    tr_vertex_coordinates[tr_curr_num_of_vertex] = intersectionPoint2
                    tr_vertex_ws[tr_curr_num_of_vertex] = intersectionPoint2W
                    tr_vertices_set.append(tr_curr_num_of_vertex)                    
                    outcode_t2 = clipCoordinateToOutcode(intersectionPoint2, intersectionPoint2W)
                    tr_vertex_outcodes[tr_curr_num_of_vertex] =  outcode_t2
                    tr_num_of_vertices = tr_num_of_vertices+1
                    
                    tempIntersectingData = [tr_curr_num_of_vertex, edge2, outsideVertex, currPlane]
                    intersectingData[tr_curr_num_of_vertex] = tempIntersectingData
                    
                    
                    redIntvert = currImageColours[edge2][0] + prop_t2 * (currImageColours[edge2][0] - currImageColours[outsideVertex][0])
                    greenIntvert = currImageColours[edge2][1] + prop_t2 * (currImageColours[edge2][1] - currImageColours[outsideVertex][1])
                    blueIntvert = currImageColours[edge2][2] + prop_t2 * (currImageColours[edge2][2] - currImageColours[outsideVertex][2])
                                             
                    currImageColours[tr_curr_num_of_vertex] = [redIntvert, greenIntvert, blueIntvert]
                    
                    tr_curr_num_of_vertex += 1
                    
                   
             
                    edges.append([tr_curr_num_of_vertex-2,edge1])
                    edges.append([tr_curr_num_of_vertex-1,edge2])
                    edges.append([tr_curr_num_of_vertex-1,tr_curr_num_of_vertex-2])

                elif num_of_outside_vertices == 2:
                    
                    
                    outsideVertex1 = tr_outside_vertex_set[0]
                    outsideVertex2 = tr_outside_vertex_set[1]
                    
                    
                    
                    edge1_1, edge1_2 =  findEdges(outsideVertex1)

                    insidevertexOfoutside1 = 0
                    outsidevertexOfOutside1 = 0
                    
                    if(edge1_1 != outsideVertex2):
                        insidevertexOfoutside1 = edge1_1
                    else:
                        insidevertexOfoutside1 = edge1_2
                    
                    if(edge1_1 != insidevertexOfoutside1):
                        outsidevertexOfOutside1 = edge1_1
                    else:
                        outsidevertexOfOutside1 = edge1_2
                   
                   
                   
                    insideVertex1_cord = tr_vertex_coordinates[insidevertexOfoutside1]
                    insideVertex1W = tr_vertex_ws[insidevertexOfoutside1]                    
                    outsideVertex1_cord = tr_vertex_coordinates[outsideVertex1]
                    outsideVertex1W =tr_vertex_ws[outsideVertex1]
                    
                    prop_t1, intersectionPoint1, intersectionPoint1W = cpp_vertexPlaneIntersectionPoint(insideVertex1_cord, outsideVertex1_cord, 
                                                                                        insideVertex1W, outsideVertex1W, currPlane)

                    removeEdges(outsideVertex1)
            
                    tr_vertices_set.remove(outsideVertex1)
                    tr_num_of_vertices = tr_num_of_vertices-1
                    
                    
                    
                    if currPlane == 5:
                        intersectionPoint1W = -intersectionPoint1W
                        
                    
                    tr_vertex_coordinates[tr_curr_num_of_vertex] = intersectionPoint1
                    tr_vertex_ws[tr_curr_num_of_vertex] = intersectionPoint1W
                    tr_vertices_set.append(tr_curr_num_of_vertex)
                    tr_num_of_vertices = tr_num_of_vertices+1
                   
                    outcode_t1 = clipCoordinateToOutcode(intersectionPoint1,  intersectionPoint1W)
                    tr_vertex_outcodes[tr_curr_num_of_vertex] =  outcode_t1
                    
                    tempIntersectingData = [tr_curr_num_of_vertex, insidevertexOfoutside1, outsideVertex1, currPlane]
                    intersectingData[tr_curr_num_of_vertex] = tempIntersectingData
                    
                    
                    redIntvert = currImageColours[insidevertexOfoutside1][0] + prop_t1 * (currImageColours[insidevertexOfoutside1][0] - currImageColours[outsideVertex1][0])
                    greenIntvert = currImageColours[insidevertexOfoutside1][1] + prop_t1 * (currImageColours[insidevertexOfoutside1][1] - currImageColours[outsideVertex1][1])
                    blueIntvert = currImageColours[insidevertexOfoutside1][2] + prop_t1 * (currImageColours[insidevertexOfoutside1][2] - currImageColours[outsideVertex1][2])
                                             
                    currImageColours[tr_curr_num_of_vertex] = [redIntvert, greenIntvert, blueIntvert]
                    
                    
                    tr_curr_num_of_vertex += 1
                    
                    edges.append([tr_curr_num_of_vertex-1,insidevertexOfoutside1])
                    edges.append([tr_curr_num_of_vertex-1,outsidevertexOfOutside1])

                    
                    
                    edge2_1, edge2_2 =  findEdges(outsideVertex2)
                    
                    
                    insidevertexOfoutside2 = 0
                    outsidevertexOfOutside2 = 0
                    
                    if(edge2_1 != outsideVertex1):
                        insidevertexOfoutside2 = edge2_1
                    else:
                        insidevertexOfoutside2 = edge2_2
                        
                    if(edge2_1 != insidevertexOfoutside2):
                        outsidevertexOfOutside2 = edge2_1
                    else:
                        outsidevertexOfOutside2 = edge2_2
                    
                    
                    insideVertex2_cord = tr_vertex_coordinates[insidevertexOfoutside2]
                    insideVertex2W = tr_vertex_ws[insidevertexOfoutside2]
                    
                    outsideVertex2_cord = tr_vertex_coordinates[outsideVertex2]
                    outsideVertex2W = tr_vertex_ws[outsideVertex2]
                    
                    prop_t2, intersectionPoint2, intersectionPoint2W = cpp_vertexPlaneIntersectionPoint(insideVertex2_cord, outsideVertex2_cord,
                                                                                                        insideVertex2W,outsideVertex2W, currPlane)
                    
                    removeEdges(outsideVertex2)      
                    tr_vertices_set.remove(outsideVertex2)
                    tr_num_of_vertices = tr_num_of_vertices-1
                    
                    if currPlane == 5:
                        intersectionPoint2W = -intersectionPoint2W
                    
                    tr_vertex_coordinates[tr_curr_num_of_vertex] = intersectionPoint2
                    tr_vertex_ws[tr_curr_num_of_vertex] = intersectionPoint2W
                    tr_vertices_set.append(tr_curr_num_of_vertex)
                    tr_num_of_vertices = tr_num_of_vertices+1
                    outcode_t1 = clipCoordinateToOutcode(intersectionPoint2,  intersectionPoint2W)
                    tr_vertex_outcodes[tr_curr_num_of_vertex] =  outcode_t1
                    
                    tempIntersectingData = [tr_curr_num_of_vertex, insidevertexOfoutside2, outsideVertex2, currPlane]
                    intersectingData[tr_curr_num_of_vertex] = tempIntersectingData
                    
                    
                    redIntvert = currImageColours[insidevertexOfoutside2][0] + prop_t2 * (currImageColours[insidevertexOfoutside2][0] - currImageColours[outsideVertex2][0])
                    greenIntvert = currImageColours[insidevertexOfoutside2][1] + prop_t2 * (currImageColours[insidevertexOfoutside2][1] - currImageColours[outsideVertex2][1])
                    blueIntvert = currImageColours[insidevertexOfoutside2][2] + prop_t2 * (currImageColours[insidevertexOfoutside2][2] - currImageColours[outsideVertex2][2])
                                             
                    currImageColours[tr_curr_num_of_vertex] = [redIntvert, greenIntvert, blueIntvert]
                    
                    
                    tr_curr_num_of_vertex += 1

                    edges.append([tr_curr_num_of_vertex-1,insidevertexOfoutside2])
                    edges.append([tr_curr_num_of_vertex-1,outsidevertexOfOutside2])  
                    
                    
                    
        if tr_num_of_vertices < 3:
            return 0     
                   

        intersectingVertices = list(set(tr_vertices_set) - set(fullyInsideVerticesNumber))
        
        
        
        set_option(rational_to_decimal=False)
        
        
        for currVert in intersectingVertices:
           
            currIntersectingData = intersectingData[currVert]
           
            
            currInsideVertCoordinates =[0,0,0]
            currOutsideVertCoordinates =[0,0,0]
            
            if currIntersectingData[1] <3:
                currInsideVertCoordinates = vertices[currTriangleVertices[currIntersectingData[1]]*3:currTriangleVertices[currIntersectingData[1]]*3+3]
            else:   
                tempCoordinates = outValueToWorldCoordinates(tr_vertex_coordinates[currIntersectingData[1]])
                currInsideVertCoordinates = [tempCoordinates[0]+posXp, tempCoordinates[1]+posYp, tempCoordinates[2]+posZp]
            
            if currIntersectingData[2] <3:
                currOutsideVertCoordinates = vertices[currTriangleVertices[currIntersectingData[2]]*3:currTriangleVertices[currIntersectingData[2]]*3+3]
            else:
                tempCoordinates = outValueToWorldCoordinates(tr_vertex_coordinates[currIntersectingData[2]])
                currOutsideVertCoordinates = [tempCoordinates[0]+posXp, tempCoordinates[1]+posYp, tempCoordinates[2]+posZp]
        
            currentIntersectingPlane = currIntersectingData[3]
            
            isIntersect, vertexPixelValue2,intersectionPoint,mp,mq =planeEdgeIntersectionUpdated(currentIntersectingPlane,currInsideVertCoordinates, currOutsideVertCoordinates,m,1)
                   
            
            
            if( isIntersect== 1):
                

                fflag =0
                currIntersectionData = []
                for vPixel in range(0, len(vertexPixelValue2),2):
                    # print("vPixel = ", vPixel)
                    if( (vertexPixelValue2[vPixel]>=0 and vertexPixelValue2[vPixel]<=49) and 
                        (vertexPixelValue2[vPixel+1]>=0 and vertexPixelValue2[vPixel+1]<=49)
                        ) : 
                        
                        if fflag == 0:
                            # numberOfIntersectingPlanes +=1
                            numberOfIntersectingEdges += 1
                            fflag = 1
                        
                        xpixel = vertexPixelValue2[vPixel]
                        ypixel = vertexPixelValue2[vPixel+1]    
                        
                        if((currIntersectingData[1] >2 and currIntersectingData[2] >2) and ((xpixel == 0 and ypixel ==0) or(xpixel ==48 and ypixel==0) or
                                                                                            (xpixel == 0 and ypixel ==48) or(xpixel ==48 and ypixel==48))):
                            
                            singleIntersectingData = [-2, currInsideVertCoordinates,currOutsideVertCoordinates,currentIntersectingPlane,\
                                                xpixel,ypixel, currIntersectingData[1], currIntersectingData[2], currIntersectingData[0]] 
                        else:                                                      
                            singleIntersectingData = [-1, currInsideVertCoordinates,currOutsideVertCoordinates,currentIntersectingPlane,\
                                                xpixel,ypixel, currIntersectingData[1], currIntersectingData[2], currIntersectingData[0]]                            
                        intersectingEdgeDataToPPL.append(singleIntersectingData) 
                        currIntersectionData.append(singleIntersectingData)  
                                                 
                        currImage.append([xpixel,ypixel])
                if fflag == 1:
                    globalIntersectingVertexDataToPPL.append(currIntersectionData)
            
        currImageName = currGroupName+str(numberOfInvRegions)
    
        
        
        combinedDataToPPL = globalInsideVertexDataToPPL+ globalIntersectingVertexDataToPPL
        

        
        allImages.append(currImage)
        
        
        fA =0
        tempRegionCons = []
        tempConsString = ""
        dataToComputeDepth = []
        
        howmanyData =0
        for k in itertools.product(*combinedDataToPPL):
            howmanyData +=1

            # print("\n",howmanyData," = ", k)
        

        currDataSet = 0
        for l in itertools.product(*combinedDataToPPL):

            environment.printLog(l)
            
            dataToComputeDepth = l
           
            currDataSet +=1

            insideVertexDetailsToPPL = l[0:numberOfFullyInsideVertices]
            intersectingEdgeDataToPPL = l[numberOfFullyInsideVertices:]

           
            currImageSetConStringPolyhedra = pyparma_posInvRegion40.computeRegion(currGroupName,posZp,numberOfFullyInsideVertices,insideVertexDetailsToPPL,\
            numberOfIntersectingEdges,intersectingEdgeDataToPPL,posXp,posYp,posZp,m[xp0],m[yp0],m[zp0], outcodeP0,currImageName)
        
            
            # print("inv region generated")
            currImageSetConString = str(currImageSetConStringPolyhedra.minimized_constraints())
            currImageSetConString = currImageSetConString.replace("x0","xp0")
            currImageSetConString = currImageSetConString.replace("x1","yp0")
            currImageSetConString = currImageSetConString.replace("x2","zp0")
            currImageSetConString = currImageSetConString.replace(" = ","==")
            currImageSetConString = currImageSetConString.replace("Constraint_System {"," ")
            currImageSetConString = currImageSetConString.replace("}"," ")
            
            if(str(currImageSetConString).replace(" ","") == "-1==0" or str(currImageSetConString).replace(" ","") == "0==-1"):
               
                # sleep(1)
                continue
            tempConsString = currImageSetConString
            # fA = 1
            currImageSetConString = "And("+str(currImageSetConString)+")"
            currImageSetCons = eval(currImageSetConString)
       
            scheck2 = Solver()
            scheck2.add(currImageSetCons)
            scheck2.add(currGroupRegionCons)
            # # scheck.add(z3invRegionCons)
            
            # scheck.push()
            scheck2.add(And(xp0 ==m[xp0], yp0 == m[yp0], zp0 == m[zp0]))
            if(scheck2.check() != sat):                
                # sleep(3)
                continue
        
            fA = 1
            
            s2.add(Not(currImageSetCons))             
            tempRegionCons.append(currImageSetConString)             
            break    
            
       
        if fA == 0:
            

            foundR, consOfReg,minmaxDepths, centerPointImage = oldInvComputation1.computeInv( posXp, posYp, posZp, currTriangle,m)
            
            
            scheck3 = Solver()
           
            scheck3.add(eval("And("+str(consOfReg)+")"))
            # scheck3.add(currGroupRegionCons)
            # scheck.add(z3invRegionCons)            
            
            # sleep(5)
            scheck3.add(And(xp0 ==m[xp0], yp0 == m[yp0], zp0 == m[zp0]))

            if(scheck3.check() == sat and foundR == 1 ):
                currImageSetCons = eval("And("+str(consOfReg)+")")
                s2.add(Not(currImageSetCons))  
                numberOfInvRegions +=1                
                intervalImageFunctions1.updateSingleIntervalImage(currTriangleIntervalImage,  currTriangle, minmaxDepths, centerPointImage)
                
                s2.add(notTheCurrentPosCons1)

            else:                
                
                eFile = open("ErrorLog.txt","a")
                eFile.write("Error in finding the region for the triangle "+str(currTriangle)+"\n")
                eFile.write("Current position = "+str(posXp)+", "+str(posYp)+", "+str(posZp)+"\n")
                eFile.write("Current consOfReg = "+str(consOfReg)+"\n")
                eFile.close()

                nmBound = 0.0001
               
                
                consOfReg2 =   str(posXp-nmBound) +"<=xp0, xp0<="+ str(posXp+nmBound)+"," \
                    + str(posYp-nmBound) +"<=yp0, yp0<="+ str(posYp+nmBound)+"," \
                    + str(posZp-nmBound) +"<=zp0, zp0<="+ str(posZp+nmBound)
                
                currImageSetCons = eval("And("+str(consOfReg2)+")")
                
                s2.add(Not(currImageSetCons))  
                intervalImageFunctions1.updateSingleIntervalImage(currTriangleIntervalImage,  currTriangle, minmaxDepths, centerPointImage)
                

            
       
            
            set_param('parallel.enable', True)
            s2.set("sat.local_search_threads", 28)
            s2.set("sat.threads", 28)
            
            
            break

             
        currImageSetConString = tempConsString
        
        
        
         
        set_option(rational_to_decimal=True)
       
        
        environment.imagePos[currImageName] = str(posXp)+" "+str(posYp)+" "+str(posZp)
       
        environment.imagePos[currImageName] = str(posXp)+" "+str(posYp)+" "+str(posZp)
       
        
        currTriangleInvPositions[numberOfInvRegions*3+0] = str(posXp)
        currTriangleInvPositions[numberOfInvRegions*3+1] = str(posYp)
        currTriangleInvPositions[numberOfInvRegions*3+2] = str(posZp)
        
        numberOfInvRegions +=1
        
        if numberOfInvRegions > 1000:            
            # print("Current Triangle = "+str(currTriangle))
            eFile = open("ErrorLog.txt","a")
            eFile.write("more that 1000 inv regions for "+str(currTriangle)+"\n")
            eFile.write("Current position = "+str(posXp)+", "+str(posYp)+", "+str(posZp)+"\n")
            eFile.write("Current consOfReg = "+str(consOfReg)+"\n")
            eFile.close()

            s2.add(Not(currGroupRegionCons))
            # exit()
            # break
            # sleep(1)
     
        s2.add(notTheCurrentPosCons1)
       
            
        set_param('parallel.enable', True)
        s2.set("sat.local_search_threads", 28)
        s2.set("sat.threads", 28)
        # sleep(3)
 
    if numberOfInvRegions <= 0:
        return 0              
    
    currIntervalImageToStore = dict()
    currIntervalImageToStore.clear()
    backGroundData = [1,25,24,2000,2000]
    
    for currPixel, currPixelData in currTriangleIntervalImage.items():
        
        if(len(currPixelData) < numberOfInvRegions ):
            currPixelData.append(backGroundData)
            
            
        rmin = min(x[0] for x in currPixelData)
        rmax = max(x[0] for x in currPixelData)
        
        gmin = min(x[1] for x in currPixelData)
        gmax = max(x[1] for x in currPixelData)
        
        bmin = min(x[2] for x in currPixelData)
        bmax = max(x[2] for x in currPixelData)
        
        dmin = min(x[3] for x in currPixelData)
        dmax = max(x[4] for x in currPixelData)
        
    
        currIntervalImageToStore[currPixel] = [rmin, rmax, gmin, gmax, bmin, bmax, dmin, dmax]
    dictionaryOfTriangleIntervalImages[currTriangle] = currIntervalImageToStore
    
    
    return numberOfInvRegions

    
        

    

def computePixelIntervals(currGroupName, currGroupRegionCons, fromSplitRegion=0):
   
    globalIntervalImage.clear()  
    dictionaryOfTriangleIntervalImages.clear()
    
    numberOfreg = [0]*environment.numOfTriangles
    s100 = Solver()
    s100.add(currGroupRegionCons)
    s100.check()
    m100 = s100.model()

    global posZp100
    posZp100 = (eval("m100[zp0].numerator_as_long()/m100[zp0].denominator_as_long()"))

    for i in range(0 , environment.numOfTriangles): 
        global currZ3VariableIndex
        currZ3VariableIndex = 0
        global currTriangleUniquePositions
        currTriangleUniquePositions = 0
        numberOfreg[i] = computeTriangleInvariantRegions2(i,currGroupName, currGroupRegionCons)
        
        if(numberOfreg[i] > 0):
            updateGlobalIntervalImage2(i,numberOfreg[i])
        
    
    prepareFinalIntervalImage(globalIntervalImage)
    
    generateVnnlbPropertyfile.generate_vnnlib_files2(finalGlobalIntervalImage)
    
    




currAbsGroupName = "A_"
currAbsGroupRegionCons = environment.initCubeCon



computePixelIntervals( currAbsGroupName, currAbsGroupRegionCons)

print("Interval image generated. Pixel min values: globalMin.txt. Pixel max values: globalMax.txt ")