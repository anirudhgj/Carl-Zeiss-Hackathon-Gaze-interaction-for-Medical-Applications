# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:48:47 2018

@author: JACK
"""

import cv2
import numpy as np

#==============================================================================
# first=[[0,960],[400,1920]]
# second=[[0,0],[400,960]]
# third=[[400,0],[800,960]]
# fourth=[[400,960],[800,1920]]
#==============================================================================

#==============================================================================
# 
# first=[[5,765],[400,1530]]
# second=[[5,5],[400,765]]
# third=[[400,5],[800,765]]
# fourth=[[400,765],[800,1530]]
# 
# quads={0:first,1:second,2:third,3:fourth}
# 
# 
# while(1):
#     img=np.zeros([800,1530])
# 
#     rand_quad=np.random.randint(0,3)
#     x,y=np.random.randint(quads[rand_quad][0][0],quads[rand_quad][1][0]),np.random.randint(quads[rand_quad][0][1],quads[rand_quad][1][1])
# 
#     img[x-5:x+5,y-5:y+5]=255
#     print(x,y)
#     cv2.imshow('safas',img)
#     k=cv2.waitKey(0)
#     if k== ord('q'):
#         cv2.destroyAllWindows()
#         break
#     
#     
#==============================================================================
def prep():
    #==============================================================================
    # first=[[0,960],[400,1920]]
    # second=[[0,0],[400,960]]
    # third=[[400,0],[800,960]]
    # fourth=[[400,960],[800,1920]]
    #==============================================================================
    startx,starty=0,0
    endx,endy=800,1530
    
    first=[[5,765],[400,1530]]
    second=[[5,5],[400,765]]
    third=[[400,5],[800,765]]
    fourth=[[400,765],[800,1530]]
    
    quads={0:first,1:second,2:third,3:fourth}
    
    
    img=np.zeros([800,1530])

    rand_quad=np.random.randint(0,3)
    x,y=np.random.randint(quads[rand_quad][0][0],quads[rand_quad][1][0]),np.random.randint(quads[rand_quad][0][1],quads[rand_quad][1][1])

    img[x-5:x+5,y-5:y+5]=255
#    print(x,y)
#    cv2.imshow('safas',img)
#    k=cv2.waitKey(0)
#    if k== ord('q'):
#        cv2.destroyAllWindows()
#        break
    a=[img,[x,y]]
    return a
def prep_with_limit(startx,starty,endx,endy):
    #==============================================================================
    # first=[[0,960],[400,1920]]
    # second=[[0,0],[400,960]]
    # third=[[400,0],[800,960]]
    # fourth=[[400,960],[800,1920]]
    #==============================================================================
    
    
#    first=[[startx+5,int(endy/2)],[int(endx/2),endy]]
#    second=[[startx+5,starty+5],[int(endx/2),int(endy/2)]]
#    third=[[int(endx/2),starty+5],[endx,int(endy/2)]]
#    fourth=[[int(endx/2),int(endy/2)],[endx,endy]]
    
    first=[[endx-int((endx-startx)/2),endy-int((endy-starty)/2)],[endx-int((endx-startx)/2),endy-int((endy-starty)/2)]]
    second=[[startx,starty],[endx-int((endx-startx)/2),endy-int((endy-starty)/2)]]
    third=[[endx-int((endx-startx)/2),starty],[endx,endy-int((endy-starty)/2)]]
    fourth=[[endx-int((endx-startx)/2),endy-int((endy-starty)/2)],[endx,endy]]
    quads={0:first,1:second,2:third,3:fourth}
    
    
    img=np.zeros([endx,endy])

    rand_quad=np.random.randint(0,3)
    x,y=np.random.randint(quads[rand_quad][0][0],quads[rand_quad][1][0]),np.random.randint(quads[rand_quad][0][1],quads[rand_quad][1][1])

    img[x-5:x+5,y-5:y+5]=255
#    print(x,y)
#    cv2.imshow('safas',img)
#    k=cv2.waitKey(0)
#    if k== ord('q'):
#        cv2.destroyAllWindows()
#        break
    a=[img,rand_quad]
    return a
