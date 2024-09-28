

from z3 import *
from pyparma import *
import anytree

import scene
import camera


global dnnOutput, imagesMap, numOfVertices, vertices, numOfTriangles, nvertices, groupFrustum,\
        initFlag,imageGroup, outFileName, initialImageCount, imagePos, imageGroupStack,groupCount,groupFrustumFlag,\
                GloballoopCount, pplInputFileName, pplOutputFileName, initialZP, pplpathHullOutputFileName, \
                        collisionCheckStartTriangle, pplSingleImageConstraintOutput,grpCubePoses,\
                                x0,x1,y0,y1,z0,z1, intiFrusCons, initCubeCon, randomLoopLimit,numOfEdges,\
                                        canvasWidth, canvasHeight, focalLength,t,b,l,r,n,f,imageCons,groupRegionConsPPL,\
                                                groupCube,allInSameGrp,groupCubeZ3, targetRegionPolyhedron, \
                                                        groupCubePostRegion,z3timeout,absStack, splitRegionPd, splitCount,\
                                                                imageWidth, imageHeight, depthOfTheInitialCube, A, numberOfSplit,\
                                                                        midPoints, processedMidPoints, spuriousCollisionData, vertColours, nnenumFlag, refineCount
dnnOutput = dict()
imagesMap = dict()
imageGroup = dict()
imageCons = dict()
imagePos = dict()
groupRegionConsPPL = dict()
groupCube = dict()
groupCubeZ3 = dict()
groupCubePostRegion =dict()

midPoints = {}
processedMidPoints = {}
spuriousCollisionData = {}

splitRegionPd =dict()
splitCount = 0
numberOfSplit = 2
# numberOfRandomPointsToCheck = 10
refineCount =0
#envs
nnenumFlag =0

A = anytree.Node("A")

groupFrustum = {}
# numOfTriangles = 4*4+250# 290+4*4 # 290 # 4+1+1
# numOfVertices = 4*6+(250*3)#+6*4# 290*3 # 6+3+3
# numOfEdges =  4*9+(250*3)#+9*4 #290*3# 9+3+3

def printLog(message):
    print(message)


#buildings2
numOfTriangles = scene.numOfTriangles# 290+4*4 # 290 # 4+1+1
numOfVertices = scene.numOfVertices#+6*4# 290*3 # 6+3+3
numOfEdges =  scene.numOfEdges#+9*4 #290*3# 9+3+3






z3timeout = 0

initFlag = 0
outFileName = "Env_11_12_8_abs_1_195_20Steps_1.txt"
initialImageCount = 0

allInSameGrp = dict()


groupCount =1
groupFrustumFlag = {}
GloballoopCount = 0
pplInputFileName = "imagesDataFromPython.txt"
pplOutputFileName = "constraintsFromPPL.txt"
pplpathHullOutputFileName = "pathHullOutput.txt"
collisionCheckStartTriangle = 4
pplSingleImageConstraintOutput = "singleImageconstraintsFromPPL.txt"

imageWidth =49
imageHeight = 49
canvasWidth = 0.9872
canvasHeight = 0.735
focalLength = 35
t =0.35820895522388063
b =-0.35820895522388063
l =-0.35820895522388063
r =0.35820895522388063


n = 1
f = 1000


grpCubePoses = dict()
imageGroupStack = []

absStack = []

xp0, yp0, zp0 = Reals('xp0 yp0 zp0')


depthOfTheInitialCube = .01


intiFrusCons = [10*xp0>=1,1000*xp0<=101,10*yp0>=45,1000*yp0<=4501, 10*zp0>=1215,1000*zp0<=121501]
initCubeCon = And(10*xp0>=1,1000*xp0<=101,10*yp0>=45,1000*yp0<=4501, 10*zp0>=1215,1000*zp0<=121501)


xp0 = Variable(0)
yp0 = Variable(1)
zp0 = Variable(2)
pd3 = NNC_Polyhedron(3)
pd3.add_constraint(10*xp0>=1)
pd3.add_constraint(1000*xp0<=101)
pd3.add_constraint(10*yp0>=45)
pd3.add_constraint(1000*yp0<=4501)
pd3.add_constraint(10*zp0>=1215)
pd3.add_constraint(1000*zp0<=121501)

midPoints["A"] = [0.1,4.5,121.5]
currentMidPoint = midPoints["A"]
currentMidPointString = str(currentMidPoint[0])+"_"+str(currentMidPoint[1])+"_"+str(currentMidPoint[2])
processedMidPoints[currentMidPointString] = "A"


groupCube["G_0"] = pd3.minimized_constraints()
groupCube["G_1"] = pd3.minimized_constraints()
groupCube["G_2"] = pd3.minimized_constraints()

groupCube["A"] = pd3.minimized_constraints()
# groupCube["A_1"] = pd3.minimized_constraints()
# groupCube["A_2"] = pd3.minimized_constraints()



groupCubeZ3["G_0"] = initCubeCon
groupCubeZ3["G_1"] = initCubeCon
groupCubeZ3["G_2"] = initCubeCon

groupCubeZ3["A"] = initCubeCon


groupCube["initCubeCon"] = pd3.minimized_constraints()
groupCubePostRegion["initCubeCon"] = pd3.minimized_constraints()
groupCubePostRegion["A"] = pd3.minimized_constraints()



z0 = 194.5
z1 = 194.51


# initialZP = z0

randomLoopLimit = 1000



vertices = scene.vertices
nvertices = scene.nvertices
vertColours = scene.vertColours
tedges = scene.tedges




