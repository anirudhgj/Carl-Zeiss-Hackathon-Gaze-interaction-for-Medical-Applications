# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:05:39 2018

@author: JACK
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 23:18:46 2018

@author: JACK
"""

import cv2
import numpy as np



cam1=cv2.VideoCapture(1)
cam1.release()
cam1=cv2.VideoCapture(1)
cam1.release()
cam1=cv2.VideoCapture(1)




try:
    while(1):
        ret,left=cam1.read()
        
        cv2.imshow('left',left)
        
        
        k=cv2.waitKey(1)
        if k==ord('q'):
            cam1.release()
            cv2.destroyAllWindows()
            break
except Exception as e:
    print(e)
    cam1.release()
    cv2.destroyAllWindows()
    