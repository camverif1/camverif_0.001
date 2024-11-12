from pyparma import *
from z3 import *
import environment
import math
from time import sleep
import numpy as np
from fractions import Fraction


def mygcd(a, b):
    if (a == 0):
        return b
    return mygcd(b % a, a)


def lcm(a, b):
    try:
        return np.lcm(a, b)
    except:
        return a*b // mygcd(a, b)


vertices = environment.vertices
xp0 = Variable(0)
yp0 = Variable(1)
zp0 = Variable(2)
vertices = environment.vertices
numOfVertices = environment.numOfVertices
tedges = environment.tedges
numOftedges = environment.numOfEdges
# intiFrusCons = environment.intiFrusCons
initCubeCon = environment.initCubeCon
canvasWidth = environment.canvasWidth
canvasHeight = environment.canvasHeight
focalLength = environment.focalLength
t = environment.t
b = environment.b
l = environment.l
r = environment.r
n = environment.n
f = environment.f

unsatFlag = 0
def addAvertexPixelConstraintIntersect(currZP, x,y,z, pixelX, pixelY, pd,planeId):
    if (isinstance(x, float)):
        # print("float x")
        xf = str(Fraction(x).limit_denominator())
        xl = xf.split('/')
        px = int(xl[0])
        # qx = int(xl[1])
        if(len(xl) == 2):
            qx = int(xl[1])
        else:
            qx = 1

    else:
        px = x
        qx = 1
    if (isinstance(y, float)):
        # print("float y")
        yf = str(Fraction(y).limit_denominator())
        yl = yf.split('/')
        py = int(yl[0])
        # qy = int(yl[1])
        if(len(yl) == 2):
            qy = int(yl[1])
        else:
            qy = 1
    else:
        py = y
        qy = 1
    if (isinstance(z, float)):
        # print("float z")
        zf = str(Fraction(z).limit_denominator())
        zl = zf.split('/')
        pz = int(zl[0])
        # qz = int(zl[1])
        if(len(zl) == 2):
            qz = int(zl[1])
        else:
            qz = 1
    else:
        pz = z
        qz = 1
    if(z - currZP < 0):
        print("z - currZP <0")
        
        # if (planeId == 0):
        # print("top")
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                            (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelX)*qx*(pz - qz*zp0)*1)*pow(10,6) )
        
        
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                        (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelX+1))*qx*(pz - qz*zp0)*1)*pow(10,6) )

        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                            (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelY)*qy*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                            (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelY+1))*qy*(pz - qz*zp0)*1)*pow(10,6) )
      
    else:
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) >= ((pixelX)*qx*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) < (((pixelX+1))*qx*(pz - qz*zp0)*1)*pow(10,6) )

        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) >= ((pixelY)*qy*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) < (((pixelY+1))*qy*(pz - qz*zp0)*1)*pow(10,6) )
    

def addAvertexPixelConstraint(currZP, vertexNumber, pixelX, pixelY, pd):
    x = vertices[vertexNumber*3+0]
    y = vertices[vertexNumber*3+1]
    z = vertices[vertexNumber*3+2]
    if (isinstance(x, float)):
        # print("float x")
        xf = str(Fraction(x).limit_denominator())
        xl = xf.split('/')
        px = int(xl[0])
        # qx = int(xl[1])
        if(len(xl) == 2):
            qx = int(xl[1])
        else:
            qx = 1

    else:
        px = x
        qx = 1
    if (isinstance(y, float)):
        # print("float y")
        yf = str(Fraction(y).limit_denominator())
        yl = yf.split('/')
        py = int(yl[0])
        # qy = int(yl[1])
        if(len(yl) == 2):
            qy = int(yl[1])
        else:
            qy = 1
    else:
        py = y
        qy = 1
    if (isinstance(z, float)):
        # print("float z")
        zf = str(Fraction(z).limit_denominator())
        zl = zf.split('/')
        pz = int(zl[0])
        # qz = int(zl[1])
        if(len(zl) == 2):
            qz = int(zl[1])
        else:
            qz = 1
    else:
        pz = z
        qz = 1
    if(z - currZP < 0):
        # print("z - currZP <0")
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                            (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelX)*qx*(pz - qz*zp0)*1)*pow(10,6) )
        
        
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelX+1))*qx*(pz - qz*zp0)*1)*pow(10,6) )

        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelY)*qy*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelY+1))*qy*(pz - qz*zp0)*1)*pow(10,6) )

    else:
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) >= ((pixelX)*qx*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) < (((pixelX+1))*qx*(pz - qz*zp0)*1)*pow(10,6) )

        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) >= ((pixelY)*qy*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) < (((pixelY+1))*qy*(pz - qz*zp0)*1)*pow(10,6) )


def getCurrentPosOutcodeCons2(outcodeP0, pd):

    mproj_1_1 = int(float((2 * n / (t - b)))*pow(10, 58)//1)

    mproj_0_0 = int(float((2 * n / (r - l)))*pow(10, 58)//1)

    mproj_2_2 = int(float((-(f + n) / (f - n)))*pow(10, 58)//1)
    mproj_3_2 = int(float((-2 * f * n / (f - n)))*pow(10, 58)//1)
    for i in range(0, numOfVertices):
        # print(i)
        x = vertices[i*3+0]
        y = vertices[i*3+1]
        z = vertices[i*3+2]
        # print(i, ": ", x, y, z)
        if (isinstance(x, float)):
            xf = str(Fraction(x).limit_denominator())
            xl = xf.split('/')
            # print(xl)
            px = int(xl[0])
            if(len(xl) == 2):
                qx = int(xl[1])
            else:
                qx = 1
        else:
            px = x
            qx = 1
        if (isinstance(y, float)):
            yf = str(Fraction(y).limit_denominator())
            yl = yf.split('/')
            py = int(yl[0])
            # qy = int(yl[1])
            if(len(yl) == 2):
                qy = int(yl[1])
            else:
                qy = 1
        else:
            py = y
            qy = 1
        if (isinstance(z, float)):
            zf = str(Fraction(z).limit_denominator())
            zl = zf.split('/')
            pz = int(zl[0])
            # qz = int(zl[1])
            if(len(zl) == 2):
                qz = int(zl[1])
            else:
                qz = 1

        else:
            pz = z
            qz = 1

        if(outcodeP0[i*6+0] == 0):
            #pd.add_constraint( (vertices[i*3+1] -yp0) * mproj_1_1   <=      -(vertices[i*3+2] -zp0)*pow(10,58) )
            pd.add_constraint(qz*(py - qy*yp0) * mproj_1_1 <= -
                              (qy)*(pz - qz*zp0)*pow(10, 58))
        else:
            # pd.add_constraint( (vertices[i*3+1] -yp0) * mproj_1_1  >   -(vertices[i*3+2] -zp0) *pow(10,58) )
            pd.add_constraint(qz*(py - qy*yp0) * mproj_1_1 > -
                              (qy)*(pz - qz*zp0) * pow(10, 58))

        if(outcodeP0[i*6+1] == 0):
            # pd.add_constraint( ((   (vertices[i*3+2] -zp0)*pow(10,58)  ) <=      (vertices[i*3+1] -yp0) * mproj_1_1 ) )
            pd.add_constraint(((qy*(pz - qz*zp0)*pow(10, 58))
                              <= qz*(py - qy*yp0) * mproj_1_1))
        else:
            # pd.add_constraint( ( (  (vertices[i*3+2] -zp0)*pow(10,58) ) >   (vertices[i*3+1] -yp0) * mproj_1_1 )  )
            pd.add_constraint(((qy*(pz - qz*zp0)*pow(10, 58))
                              > qz*(py - qy*yp0) * mproj_1_1))

        if(outcodeP0[i*6+2] == 0):
            #    pd.add_constraint( ( (vertices[i*3+0] -xp0) *mproj_0_0  <=     -(vertices[i*3+2] -zp0)*pow(10,58) ) )
            pd.add_constraint(
                (qz*(px - qx*xp0) * mproj_0_0 <= -(qx)*(pz - qz*zp0)*pow(10, 58)))
        else:
            # pd.add_constraint( ( (vertices[i*3+0] -xp0) *mproj_0_0  >    -(vertices[i*3+2] -zp0)*pow(10,58) )  )
            pd.add_constraint(
                (qz*(px - qx*xp0) * mproj_0_0 > -(qx)*(pz - qz*zp0)*pow(10, 58)))

        if(outcodeP0[i*6+3] == 0):
            # pd.add_constraint( ( (  (vertices[i*3+2] -zp0)*pow(10,58)  )<=      (vertices[i*3+0] -xp0) *mproj_0_0 ) )
            pd.add_constraint(((qx*(pz - qz*zp0)*pow(10, 58))
                              <= qz*(px - qx*xp0) * mproj_0_0))
        else:
            # pd.add_constraint( ( (  (vertices[i*3+2] -zp0)*pow(10,58)  )>   (vertices[i*3+0] -xp0) *mproj_0_0 ) )
            pd.add_constraint(((qx*(pz - qz*zp0)*pow(10, 58))
                              > qz*(px - qx*xp0) * mproj_0_0))

        if(outcodeP0[i*6+4] == 0):
            # pd.add_constraint( (  (vertices[i*3+2] -zp0) * mproj_2_2 + mproj_3_2 <=  -(vertices[i*3+2] -zp0) *pow(10,58)) )
            pd.add_constraint(((pz - qz*zp0) * mproj_2_2 +
                              qz*mproj_3_2 <= -(pz - qz*zp0) * pow(10, 58)))
        else:
            # pd.add_constraint( (  (vertices[i*3+2] -zp0) * mproj_2_2 + mproj_3_2 > -(vertices[i*3+2] -zp0) *pow(10,58) ) )
            pd.add_constraint(((pz - qz*zp0) * mproj_2_2 +
                              qz*mproj_3_2 > -(pz - qz*zp0) * pow(10, 58)))

        if(outcodeP0[i*6+5] == 0):
            # pd.add_constraint( ( (   (vertices[i*3+2] -zp0) *pow(10,58)  ) <=    (vertices[i*3+2] -zp0) * mproj_2_2 + mproj_3_2) )
            pd.add_constraint((((pz - qz*zp0) * pow(10, 58))
                              <= (pz - qz*zp0) * mproj_2_2 + qz*mproj_3_2))
        else:
            # pd.add_constraint( ( (   (vertices[i*3+2] -zp0)*pow(10,58)  ) >   (vertices[i*3+2] -zp0) * mproj_2_2 + mproj_3_2) )
            pd.add_constraint((((pz - qz*zp0)*pow(10, 58)) >
                              (pz - qz*zp0) * mproj_2_2 + qz*mproj_3_2))

def intersectingEdgeRegion(planeId, edgeId, insideVertex,outsideVertex, xPixel, yPixel, pd):
    r =0.35821;
    x1 = vertices[insideVertex*3 +0];
    y1 = vertices[insideVertex*3 +1];
    z1 = vertices[insideVertex*3 +2];

    x0 = vertices[outsideVertex*3 +0];
    y0 = vertices[outsideVertex*3 +1];
    z0 = vertices[outsideVertex*3 +2];
    
    fraction_constant = 0.01462077368
    fraction_constant_far = 14.62077368
    
    nearCorner = .35820895522388063
    farCorner = 358.20895522388063
    
    pd2 =NNC_Polyhedron(3,'empty')
    
    if planeId == 0 or planeId == 1:
        # print("top plane")
        
        #nearplane edge's top vertex
        BF_top_right_x = (x0+nearCorner)- (xPixel * fraction_constant);
        BF_top_left_x = (x0+nearCorner)- ((xPixel+1) * fraction_constant);
        #nearplane edge's bottom vertex
        BF_bot_right_x = (x1+nearCorner)- (xPixel * fraction_constant);
        BF_bot_left_x = (x1+nearCorner)- ((xPixel+1) *fraction_constant);
        BF_top_z = (z0+1);
        BF_bot_z = (z1+1);
        FF_top_right_x = (x0+farCorner)- (xPixel * fraction_constant_far);
        FF_top_left_x = (x0+farCorner)- ((xPixel+1) * fraction_constant_far);
        FF_bot_right_x = (x1+farCorner)- (xPixel * fraction_constant_far);
        FF_bot_left_x = (x1+farCorner)- ((xPixel+1) *fraction_constant_far);
        FF_top_z = (z0+1000);
        FF_bot_z = (z1+1000);
        
        if planeId == 0:
            BF_top_y = (y0-nearCorner);
            BF_bot_y = (y1-nearCorner);
            
            FF_top_y = (y0-farCorner);
            FF_bot_y = (y1-farCorner);
        else:
            BF_top_y = (y0+nearCorner);
            BF_bot_y = (y1+nearCorner);
            
            FF_top_y = (y0+farCorner);
            FF_bot_y = (y1+farCorner);

        i_BF_top_right_x = int(BF_top_right_x*pow(10,15)//1)
        i_BF_top_left_x = int( BF_top_left_x*pow(10,15)//1)
        # print("BF_top_left_x = ", BF_top_left_x," : ", i_BF_top_left_x)
        i_BF_bot_right_x = int( BF_bot_right_x*pow(10,15)//1)
        # print("BF_bot_right_x = ", BF_bot_right_x," : ", i_BF_bot_right_x)
        i_BF_bot_left_x = int( BF_bot_left_x*pow(10,15)//1)
        i_BF_top_y = int(BF_top_y*pow(10,15)//1)
        # print("BF_top_y = ", BF_top_y," : ", i_BF_top_y)
        i_BF_bot_y = int( BF_bot_y*pow(10,15)//1)
        # print("BF_bot_y = ", BF_bot_y," : ", i_BF_bot_y)
        i_BF_top_z = int( BF_top_z*pow(10,15)//1)
        # print("BF_top_z = ", BF_top_z," : ", i_BF_top_z)
        i_BF_bot_z = int( BF_bot_z *pow(10,15)//1)
        i_FF_top_right_x = int(FF_top_right_x*pow(10,15)//1)
        i_FF_top_left_x = int( FF_top_left_x*pow(10,15)//1)
        i_FF_bot_right_x =  int( FF_bot_right_x*pow(10,15)//1)
        i_FF_bot_left_x =  int( FF_bot_left_x*pow(10,15)//1)
        i_FF_top_y =  int( FF_top_y*pow(10,15)//1)
        i_FF_bot_y =  int( FF_bot_y*pow(10,15)//1)
        i_FF_top_z =  int( FF_top_z*pow(10,15)//1)
        i_FF_bot_z =  int( FF_bot_z*pow(10,15)//1)
        
        pd2.add_generator(point( i_BF_top_right_x*xp0+i_BF_top_y*yp0+i_BF_top_z*zp0 ,pow(10,15)))
        pd2.add_generator(point( i_BF_top_left_x*xp0+ i_BF_top_y*yp0+ i_BF_top_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_BF_bot_right_x*xp0+ i_BF_bot_y*yp0+ i_BF_bot_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_BF_bot_left_x*xp0+ i_BF_bot_y*yp0+ i_BF_bot_z*zp0,pow(10,15)))

        pd2.add_generator(point(i_FF_top_right_x*xp0+ i_FF_top_y*yp0+  i_FF_top_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_top_left_x*xp0+ i_FF_top_y*yp0+ i_FF_top_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_bot_right_x*xp0+ i_FF_bot_y*yp0+ i_FF_bot_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_bot_left_x*xp0+ i_FF_bot_y*yp0+ i_FF_bot_z*zp0,pow(10,15)))
        
    if planeId == 2 or planeId ==3:
        # print("right plane")
        plane0_v0 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        
        if planeId == 2:
            BF_rVertex_x = x0 - nearCorner
            BF_lVertex_x = x0 - nearCorner
            
            FF_rVertex_x = x0 - farCorner
            FF_lVertex_x = x0 - farCorner
        
        else:
            
            BF_rVertex_x = x0 + nearCorner
            BF_lVertex_x = x0 + nearCorner
            
            FF_rVertex_x = x0 + farCorner
            FF_lVertex_x = x0 + farCorner
            
        BF_rVertex_top_y = (y0-nearCorner)+ (yPixel * fraction_constant)
        BF_rVertex_bot_y = (y0-nearCorner)+ ( (yPixel+1) * fraction_constant)
        BF_lVertex_top_y = (y1-nearCorner)+ (yPixel * fraction_constant)
        BF_lVertex_bot_y = (y1-nearCorner)+ ( (yPixel+1) * fraction_constant)

        BF_rVertex_z = (z0+1);
        BF_lVertex_z = (z1+1);
        
        FF_rVertex_top_y = (y0-farCorner)+ (yPixel * fraction_constant_far)
        FF_rVertex_bot_y = (y0-farCorner)+ ( (yPixel+1) * fraction_constant_far)
        FF_lVertex_top_y = (y1-farCorner)+ (yPixel * fraction_constant_far)
        FF_lVertex_bot_y = (y1-farCorner)+ ( (yPixel+1) * fraction_constant_far)

        FF_rVertex_z = (z0+1000);
        FF_lVertex_z = (z1+1000);
        i_BF_rVertex_x = int(BF_rVertex_x*pow(10,15)//1)
        i_BF_lVertex_x = int(BF_lVertex_x*pow(10,15)//1)
        i_FF_rVertex_x = int( FF_rVertex_x*pow(10,15)//1)
        i_FF_lVertex_x = int( FF_lVertex_x*pow(10,15)//1)
        i_BF_rVertex_top_y = int( BF_rVertex_top_y*pow(10,15)//1)
        i_BF_rVertex_bot_y = int( BF_rVertex_bot_y*pow(10,15)//1)
        i_BF_lVertex_top_y = int( BF_lVertex_top_y*pow(10,15)//1)
        i_BF_lVertex_bot_y = int(BF_lVertex_bot_y*pow(10,15)//1)
        i_FF_rVertex_top_y = int( FF_rVertex_top_y*pow(10,15)//1)
        i_FF_rVertex_bot_y = int( FF_rVertex_bot_y*pow(10,15)//1)
        i_FF_lVertex_top_y = int( FF_lVertex_top_y*pow(10,15)//1)
        i_FF_lVertex_bot_y = int( FF_lVertex_bot_y*pow(10,15)//1)
        i_BF_rVertex_z = int( BF_rVertex_z*pow(10,15)//1)
        i_BF_lVertex_z = int( BF_lVertex_z*pow(10,15)//1)
        
        i_FF_rVertex_z = int( FF_rVertex_z*pow(10,15)//1)
        i_FF_lVertex_z = int( FF_lVertex_z*pow(10,15)//1)
        
        pd2.add_generator(point( i_BF_rVertex_x*xp0+i_BF_rVertex_top_y*yp0+i_BF_rVertex_z*zp0 ,pow(10,15)))
        pd2.add_generator(point( i_BF_rVertex_x*xp0+ i_BF_rVertex_bot_y*yp0+ i_BF_rVertex_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_BF_lVertex_x*xp0+ i_BF_lVertex_top_y*yp0+ i_BF_lVertex_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_BF_lVertex_x*xp0+ i_BF_lVertex_bot_y*yp0+ i_BF_lVertex_z*zp0,pow(10,15)))

        pd2.add_generator(point( i_FF_rVertex_x*xp0+i_FF_rVertex_top_y*yp0+i_FF_rVertex_z*zp0 ,pow(10,15)))
        pd2.add_generator(point( i_FF_rVertex_x*xp0+ i_FF_rVertex_bot_y*yp0+ i_FF_rVertex_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_lVertex_x*xp0+ i_FF_lVertex_top_y*yp0+ i_FF_lVertex_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_lVertex_x*xp0+ i_FF_lVertex_bot_y*yp0+ i_FF_lVertex_z*zp0,pow(10,15)))
  
    if planeId == 5:
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v3 = [0.35820895522388063, 0.35820895522388063, -1]

    # pd.intersection_assign(pd2);
    return pd2

def findPosUpdated(x0, y0, z0, nearFar, xpixel, ypixel, planeId,intersectionPoint=0):
    
    s = Solver()
    set_option(rational_to_decimal=True)
    set_option(precision=105)
    s.set("timeout", 10000)

    xp, yp, zp = Reals('xp yp zp')
    a, b = Reals('a b')

    xk, yk, zk = Reals('xk yk zk')
    u, v, w, g = Reals('u v w g')    
    c,d = Reals('c d')
    
    tpc, bpc, rpc, lpc = Reals('tpc bpc rpc lpc') 
    
    s.add(u+v == 1)
    s.add(And(u >= 0, v >= 0))
    # s.add(And(b >= 0, b <= 0.01))

    plane0_v0 = [0, 0.0, -1]
    plane0_v1 = [0.0, 0.0, -1]
    plane0_v2 = [0, 0, -1000]
    plane0_v3 = [0, 0, -1000]

    cons1 = ""
    cons2 = ""

    if planeId == 0:
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, 358.20895522388063, -1000]
        
    if planeId == 1:
        plane0_v0 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, -358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
    if planeId == 2:
        # print("right plane")
        plane0_v0 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        
    if planeId == 3:
        print("left plane ")
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [-358.20895522388063, -358.20895522388063, -1000]
        
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
    
    if(nearFar == 1):
        s.add(x0-xp == (u*px0+(1-u)*px1))
        s.add(y0-yp == (u*py0+(1-u)*py1))
        s.add(z0-zp == (u*pz0+(1-u)*pz1))      
        
    elif(nearFar ==1000):
        s.add(x0-xp == (u*px2+v*px3))
        s.add(y0-yp == (u*py2+v*py3))
        s.add(z0-zp == (u*pz2+v*pz3))
        
    else:
      
        exit()
    
    if planeId == 0:
       
        s3 = Solver()        
        s3.add(u+v == 1)
        s3.add(And(u >= 0, v >= 0))   
        if(nearFar == 1):
            s3.add(x0-xp == (u*px0+(1-u)*px1))
            s3.add(y0-yp == (u*py0+(1-u)*py1))
            s3.add(z0-zp == (u*pz0+(1-u)*pz1))      
        
        elif(nearFar ==1000):
            s3.add(x0-xp == (u*px2+v*px3))
            s3.add(y0-yp == (u*py2+v*py3))
            s3.add(z0-zp == (u*pz2+v*pz3))
        
        newYPixel = Real('newYPixel')
        cons20 = "newYPixel == ((68.39567*(y0-yp))/(z0-zp))+24.5" 
        s3.add(eval(cons20))        
        # print(s3.check())
        m3= s3.model()
        # print(m3)
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zp))+24.5"
        cons2 = "m3[newYPixel] == ((68.39567*(y0-yp))/(z0-zp))+24.5"       
        s.add((And(eval(cons1),eval(cons2))) )    
        # print(s.check())  
        
    elif planeId == 1:
        s3 = Solver()        
        s3.add(u+v == 1)
        s3.add(And(u >= 0, v >= 0))   
        if(nearFar == 1):
            s3.add(x0-xp == (u*px0+v*px1))
            s3.add(y0-yp == (u*py0+v*py1))
            s3.add(z0-zp == (u*pz0+v*pz1))      
        
        elif(nearFar ==1000):
            s3.add(x0-xp == (u*px2+v*px3))
            s3.add(y0-yp == (u*py2+v*py3))
            s3.add(z0-zp == (u*pz2+v*pz3))
        
        newYPixel = Real('newYPixel')
        cons20 = "newYPixel == ((68.39567*(y0-yp))/(z0-zp))+24.5" 
        s3.add(eval(cons20))        
        # print(s3.check())
        m3= s3.model()
        cons1 = "xpixel == ((-68.3956*(x0-xp))/(z0-zp))+24.5"
        cons2 = "m3[newYPixel] == ((68.39567*(y0-yp))/(z0-zp))+24.5"       
        s.add((And(eval(cons1),eval(cons2))) )
        
    elif planeId == 2:
        s3 = Solver()        
        s3.add(u+v == 1)
        s3.add(And(u >= 0, v >= 0))   
        if(nearFar == 1):
            s3.add(x0-xp == (u*px0+v*px1))
            s3.add(y0-yp == (u*py0+v*py1))
            s3.add(z0-zp == (u*pz0+v*pz1))      
        
        elif(nearFar ==1000):
            s3.add(x0-xp == (u*px2+v*px3))
            s3.add(y0-yp == (u*py2+v*py3))
            s3.add(z0-zp == (u*pz2+v*pz3))
        
        newXPixel = Real('newXPixel')
        cons20 = "newXPixel == ((-68.39567*(x0-xp))/(z0-zp))+24.5" 
        s3.add(eval(cons20))        
        # print(s3.check())
        m3= s3.model()
        # print(m3)
        
        cons1 = "m3[newXPixel] == ((-68.39567*(x0-xp))/(z0-zp))+24.5"
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zp))+24.5"       
        s.add((And(eval(cons1),eval(cons2))) )        
        
    elif planeId == 3:
        print("left")
      
        s3 = Solver()        
        s3.add(u+v == 1)
        s3.add(And(u >= 0, v >= 0))   
        if(nearFar == 1):
            s3.add(x0-xp == (u*px0+v*px1))
            s3.add(y0-yp == (u*py0+v*py1))
            s3.add(z0-zp == (u*pz0+v*pz1))      
        
        elif(nearFar ==1000):
            s3.add(x0-xp == (u*px2+v*px3))
            s3.add(y0-yp == (u*py2+v*py3))
            s3.add(z0-zp == (u*pz2+v*pz3))
        
        newXPixel = Real('newXPixel')
        cons20 = "newXPixel == ((-68.39567*(x0-xp))/(z0-zp))+24.5" 
        s3.add(eval(cons20))        
        # print(s3.check())
        m3= s3.model()
        print(m3)
        cons1 = "m3[newXPixel] == ((-68.39567*(x0-xp))/(z0-zp))+24.5"
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zp))+24.5"       
        s.add((And(eval(cons1),eval(cons2))) )

    m= s.model()
    result = s.check()
    # print(result)
    if result == sat:
        m = s.model()
        posx = str(m[xp]).replace("?","")
        posy = str(m[yp]).replace("?","")
        posz = str(m[zp]).replace("?","")
        del(s)
        return [posx, posy, posz]
    if result == unknown:
        # print("timeout occured trying again")
        del(s)
        sleep(10)
        findPos(x0, y0, z0, zpos, xpixel, ypixel, planeId)
    else:
        del(s)
        global unsatFlag
        unsatFlag = 1
        exit()
        sleep(3)
        return 0

def findPos(x0, y0, z0, zpos, xpixel, ypixel, planeId,intersectionPoint=0):

    s = Solver()
    set_option(rational_to_decimal=True)
    set_option(precision=105)
    s.set("timeout", 10000)
    xp, yp, zp = Reals('xp yp zp')
    b = Real('b')
    xk, yk, zk = Reals('xk yk zk')
    u, v, w, g = Reals('u v w g')
    s.add(And(b >= 0, b <= .00001))
    s.add(u+v+w+g == 1)
    s.add(And(u >= 0, v >= 0, w >= 0, g >= 0))
    plane0_v0 = [0, 0.0, -1]
    plane0_v1 = [0.0, 0.0, -1]
    plane0_v2 = [0, 0, -1000]
    plane0_v3 = [0, 0, -1000]

    cons1 = ""
    cons2 = ""
    if planeId == 0:
        # print("top plane")
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, 358.20895522388063, -1000]
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        if ypixel != 0:
            # print("ypixel not zero")
            cons2 = "0-b == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
        else:
            cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"

    if planeId == 1:
       
        plane0_v0 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, -358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
       
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        # cons2 = "48-b == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
        if ypixel != 48:
            # print("ypixel not 48")
            cons2 = "48.00000000000003-b == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
        else:
            cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"

    if planeId == 2:
        
        plane0_v0 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        # cons1 = "48-b == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
    if planeId == 3:
        
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [-358.20895522388063, -358.20895522388063, -1000]
       
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
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

    if(intersectionPoint == 1):
        # print("Intersection point, already in the camera coordinate system")
        s.add(x0 == (u*px0+v*px1+w*px2+g*px3))
        s.add(y0 == (u*py0+v*py1+w*py2+g*py3))
        s.add(z0 == (u*pz0+v*pz1+w*pz2+g*pz3))
        cons1 = "xpixel == ((-68.39567*(x0))/(z0))+24.5"
        cons2 = "ypixel == ((68.39567*(y0))/(z0))+24.5"
        s.add(eval(cons1))
        s.add(eval(cons2))

    else:
        s.add(x0-xp == (u*px0+v*px1+w*px2+g*px3))
        s.add(y0-yp == (u*py0+v*py1+w*py2+g*py3))
        s.add(z0-zpos == (u*pz0+v*pz1+w*pz2+g*pz3))

    m= s.model()
   
    s.add(eval(cons1))
    # print(s.check())
    s.add(eval(cons2))
   
    result = s.check()
    if result == sat:
        m = s.model()
        posx = m[xp]
        posy = m[yp]
        # posz = m[zp]
        posz = zpos
        del(s)
        return [posx, posy, posz]
    if result == unknown:
        # print("timeout occured trying again")
        del(s)
        sleep(10)
        findPos(x0, y0, z0, zpos, xpixel, ypixel, planeId)
    else:
        del(s)
        global unsatFlag
        unsatFlag = 1
        exit()
        sleep(3)
        return 0

dummyGroupPolyhedra=NNC_Polyhedron(3)
dummyPolyhedraCons = dummyGroupPolyhedra.minimized_constraints()


def computeEightPointsRegion(planeId, edgeId, insideVertex, outsideVertex, xpixel, ypixel, pd, mxp, myp, mzp, ix, iy, iz, mp, mq,
                              posXp, posYp, posZp):
    
    x1 = vertices[outsideVertex*3 + 0]
    y1 = vertices[outsideVertex*3 + 1]
    z1 = vertices[outsideVertex*3 + 2]

    x0 = vertices[insideVertex*3 + 0]
    y0 = vertices[insideVertex*3 + 1]
    z0 = vertices[insideVertex*3 + 2]
    pointsList = []
    near = 1
    far =1000
    pd2 = NNC_Polyhedron(3, 'empty')
    # print(x0,y0,z0,x1,y1,z1)
    if planeId == 0:
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0 - nearRight
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1 - nearRight
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*farPart
        ypos0 = y0 - farRight
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*farPart
        ypos0 = y1 - farRight
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
    if planeId == 1:
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0 + nearRight
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1 + nearRight
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*farPart
        ypos0 = y0 + farRight
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*farPart
        ypos0 = y1 + farRight
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
    
    if planeId == 2:
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0 - nearRight
        ypos0 = y0+ (ypixel-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 - nearRight
        ypos0 =  y1+(ypixel-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0                            
        tempList[1] = y1+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0 - farRight
        ypos0 = y0+(ypixel-24.5)*farPart        
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 - farRight 
        ypos0 = y1+(ypixel-24.5)*farPart
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] =  y1+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)
    
    if planeId == 3:
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0 + nearRight
        ypos0 = y0+ (ypixel-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 + nearRight
        ypos0 =  y1+(ypixel-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0                            
        tempList[1] = y1+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0 + farRight
        ypos0 = y0+(ypixel-24.5)*farPart        
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 + farRight 
        ypos0 = y1+(ypixel-24.5)*farPart
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] =  y1+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)  
    
    if planeId == 5:
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0+ (ypixel-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*nearPart
        tempList[1] = y0+(ypixel-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0+ (ypixel+1-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel+1-24.5)*nearPart
        ypos0 = y0+ (ypixel+1-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1+ (ypixel-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*nearPart
        tempList[1] = y1+(ypixel-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1+ (ypixel+1-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel+1-24.5)*nearPart
        ypos0 = y1+ (ypixel+1-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
    for i in [0,1,3,2,4,5,7,6]:
        currPos = pointsList[i]
        # print("currPos")
        mxp1 = currPos[0]
        myp1 = currPos[1]
        mzp1 = currPos[2]

        mxp1 = int(float(mxp1)*pow(10, 16))
        myp1 = int(float(myp1)*pow(10, 16))
        mzp1 = int(float(mzp1)*pow(10, 16))

        pd2.add_generator(point(mxp1*xp0+myp1 * yp0+mzp1*zp0, pow(10, 16)))
        
    # print(pd2.minimized_constraints())
    return pd2
  

def computeEightPointsRegion2(planeId, insideVertex, outsideVertex, xpixel, ypixel):
    
    x1 =    outsideVertex[0]
    y1 =    outsideVertex[1]
    z1 =    outsideVertex[2]

    x0 =    insideVertex[0]
    y0 =    insideVertex[1]
    z0 =    insideVertex[2]
    pointsList = []
    near = 1
    far =1000
    
    pd2 = NNC_Polyhedron(3, 'empty')
    
    if planeId == 0:
        nearRight = 0.3582125714285714
        farRight = nearRight*1000
    
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0 - nearRight
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1 - nearRight
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*farPart
        ypos0 = y0 - farRight
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*farPart
        ypos0 = y1 - farRight
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
    
    if planeId == 1:
        nearRight = 0.3582125714285714
        farRight = nearRight*1000
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0 + nearRight
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1 + nearRight
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*farPart
        ypos0 = y0 + farRight
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*farPart
        ypos0 = y1 + farRight
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
    
    if planeId == 2:
        nearRight = 0.358212571428571355422244897959182
        farRight = nearRight*1000
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0 - nearRight
        ypos0 = y0+ (ypixel-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 - nearRight
        ypos0 =  y1+(ypixel-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0                            
        tempList[1] = y1+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0 - farRight
        ypos0 = y0+(ypixel-24.5)*farPart        
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 - farRight 
        ypos0 = y1+(ypixel-24.5)*farPart
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] =  y1+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)
    if planeId == 3:
        nearRight = 0.358212571428571355422244897959182
        farRight = nearRight*1000
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0 + nearRight
        ypos0 = y0+ (ypixel-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 + nearRight
        ypos0 =  y1+(ypixel-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0                            
        tempList[1] = y1+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0 + farRight
        ypos0 = y0+(ypixel-24.5)*farPart        
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 + farRight 
        ypos0 = y1+(ypixel-24.5)*farPart
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] =  y1+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)  
    
    if planeId == 5:
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0+ (ypixel-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*nearPart
        tempList[1] = y0+(ypixel-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0+ (ypixel+1-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel+1-24.5)*nearPart
        ypos0 = y0+ (ypixel+1-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1+ (ypixel-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*nearPart
        tempList[1] = y1+(ypixel-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1+ (ypixel+1-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel+1-24.5)*nearPart
        ypos0 = y1+ (ypixel+1-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
    
    # set_option(rational_to_decimal=True)
    for i in [0,1,3,2,4,5,7,6]:
        currPos = pointsList[i]
        mxp1 = currPos[0]
        myp1 = currPos[1]
        mzp1 = currPos[2]
        mxp1 = int(float(mxp1)*pow(10, 16))
        myp1 = int(float(myp1)*pow(10, 16))
        mzp1 = int(float(mzp1)*pow(10, 16))

        pd2.add_generator(point(mxp1*xp0+myp1 * yp0+mzp1*zp0, pow(10, 16)))
        
    return pd2
    
  

def computeRegion(currGroupName, currZP, numberOfFullyInsideVertices, insideVertexDetailsToPPL, numberOfIntersectingEdges,
                  intersectingEdgeDataToPPL, posXp1, posYp1, posZp1, mxp, myp, mzp, outcodeP0, currImageName, currGroupPolyhedra=dummyPolyhedraCons):
    mxpOg = mxp
    mypOg = myp
    mzpOg = mzp
    # print("\n")
    imageFrustumPolyhedron = NNC_Polyhedron(3)
    currImageCube_ph = NNC_Polyhedron(3)
    
    currImageName.startswith("split")

    if(currImageName != "singlePosImage" and (not currImageName.startswith("split"))):
        # print("pos from original computation")
        pathLength = currImageName.count('_')
        currImage_cubeName = ""

        if(pathLength == 1):
            currImage_cubeName = "initCubeCon"
        else:
            currImage_cubeName = currImageName[0:currImageName.rfind("_")]
       
        currImageCube_ph.add_constraints(
            environment.groupCubePostRegion[currImage_cubeName])
       
    elif (currImageName == "singlePosImage"):
        currImageCube_ph.add_constraints(currGroupPolyhedra)
    else:
        currImageCube_ph.add_constraints(environment.splitRegionPd["split_"+str(environment.splitCount)])
    imageFrustumPolyhedron.intersection_assign(currImageCube_ph)
    for i in range(0, numberOfFullyInsideVertices):
        currentVertexIndex = insideVertexDetailsToPPL[i][0]
        xpixel = insideVertexDetailsToPPL[i][1]
        ypixel = insideVertexDetailsToPPL[i][2]
        
        addAvertexPixelConstraint(
            currZP, currentVertexIndex, xpixel, ypixel, imageFrustumPolyhedron)

    pdC = NNC_Polyhedron(3)
    pdC = imageFrustumPolyhedron
    for i in range(0, numberOfIntersectingEdges):
        edgeId = intersectingEdgeDataToPPL[i][0]
        if edgeId != -1 and edgeId != -2  and edgeId != -3:
            insideVertex = intersectingEdgeDataToPPL[i][1]
            outsideVertex = intersectingEdgeDataToPPL[i][2]

            planeId = eval(str(intersectingEdgeDataToPPL[i][3]))
            xpixel = eval(str(intersectingEdgeDataToPPL[i][4]))
            ypixel = eval(str(intersectingEdgeDataToPPL[i][5]))

            ix = intersectingEdgeDataToPPL[i][6]
            iy = intersectingEdgeDataToPPL[i][7]
            iz = intersectingEdgeDataToPPL[i][8]

            mp = intersectingEdgeDataToPPL[i][9]
            mq = intersectingEdgeDataToPPL[i][10]

            pd2 = NNC_Polyhedron(3, 'empty')
            pd2 = computeEightPointsRegion(planeId, edgeId, insideVertex, outsideVertex, xpixel, ypixel, imageFrustumPolyhedron,
                                            mxp, myp, mzp, ix, iy, iz, mp, mq, posXp1, posYp1, posZp1)
      
            pdC.intersection_assign(pd2)
         
        elif edgeId == -1:
            pd2 = NNC_Polyhedron(3, 'empty')
            currData = intersectingEdgeDataToPPL[i]
            pd2 = computeEightPointsRegion2(currData[3], currData[1],currData[2], currData[4], currData[5])
            pdC.intersection_assign(pd2)
        elif edgeId == -2:
            pd2 = NNC_Polyhedron(3, 'empty')
            currData = intersectingEdgeDataToPPL[i]
            pd2 = computeEightPointsRegion2(currData[3], currData[1],currData[2], currData[4], currData[5])
            pdC.intersection_assign(pd2)
            
    environment.imageCons[currImageName] = pdC.minimized_constraints()
    # return str(pdC.minimized_constraints())
    return pdC


