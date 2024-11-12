from z3 import *
from pyparma import *
import anytree

import camera
import scene

global dnnOutput, imagesMap, numOfVertices, vertices, numOfTriangles, nvertices, groupFrustum,\
        initFlag,imageGroup, outFileName, initialImageCount, imagePos, imageGroupStack,groupCount,groupFrustumFlag,\
                GloballoopCount, pplInputFileName, pplOutputFileName, initialZP, pplpathHullOutputFileName, \
                        collisionCheckStartTriangle, pplSingleImageConstraintOutput,grpCubePoses,\
                                x0,x1,y0,y1,z0,z1, intiFrusCons, initCubeCon, randomLoopLimit,numOfEdges,\
                                        canvasWidth, canvasHeight, focalLength,t,b,l,r,n,f,imageCons,groupRegionConsPPL,\
                                                groupCube,allInSameGrp,groupCubeZ3, targetRegionPolyhedron, \
                                                        groupCubePostRegion,z3timeout,absStack, splitRegionPd, splitCount,\
                                                                imageWidth, imageHeight, depthOfTheInitialCube, A, numberOfSplit,\
                                                                        midPoints, processedMidPoints, spuriousCollisionData, vertColours, nnenumFlag, refineCount,\
                 initRegionMinMaxValues,initRegionCornerPoints, regionMinMaxValues, regionCornerPoints, totalNumRefinment

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

vertices = scene.vertices
numOfVertices = scene.numOfVertices
numOfTriangles = scene.numOfTriangles
nvertices = scene.nvertices
numOfEdges = scene.numOfEdges
vertColours = scene.vertColours
tedges = scene.tedges


regionMinMaxValues = {}  
regionCornerPoints = {}

totalNumRefinment =0

groupFrustum = {}

def printLog(message):
    print(message)

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

t = camera.t
b = camera.b
l = camera.l
r = camera.r
imageWidth = camera.imageWidth
imageHeight = camera.imageHeight

n = camera.nearClippingPlane
f = camera.farClippingPlane

canvasWidth = camera.filmApertureWidth
canvasHeight = camera.filmApertureHeight
focalLength = camera.focalLength


grpCubePoses = dict()
imageGroupStack = []

absStack = []

xp0, yp0, zp0 = Reals('xp0 yp0 zp0')


###################################Set Region cons here ########################################
# intiFrusCons = [10*xp0>=1,1000*xp0<=101,10*yp0>=45,1000*yp0<=4501, 10*zp0>=1215,1000*zp0<=121501]  
initCubeCon = And(10*xp0>=1,1000*xp0<=101,10*yp0>=45,1000*yp0<=4501, 10*zp0>=1215,1000*zp0<=121501)

depthOfTheInitialCube = .01 # depth of the region in centimeter

xp0 = Variable(0)
yp0 = Variable(1)
zp0 = Variable(2)
pd3 = NNC_Polyhedron(3)

#######Add same set of constraints one more time ################
pd3.add_constraint(10*xp0>=1)
pd3.add_constraint(1000*xp0<=101)
pd3.add_constraint(10*yp0>=45)
pd3.add_constraint(1000*yp0<=4501)
pd3.add_constraint(10*zp0>=1215)
pd3.add_constraint(1000*zp0<=121501)

##############################Region cons end#####################################


        
groupCube["G_0"] = pd3.minimized_constraints()
groupCube["G_1"] = pd3.minimized_constraints()
groupCube["G_2"] = pd3.minimized_constraints()
groupCube["A"] = pd3.minimized_constraints()
groupCubeZ3["G_0"] = initCubeCon
groupCubeZ3["G_1"] = initCubeCon
groupCubeZ3["G_2"] = initCubeCon
groupCubeZ3["A"] = initCubeCon
groupCube["initCubeCon"] = pd3.minimized_constraints()
groupCubePostRegion["initCubeCon"] = pd3.minimized_constraints()
groupCubePostRegion["A"] = pd3.minimized_constraints()
