# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 00:30:17 2018

@author: JACK
"""

import cv2
import numpy as np
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from pymouse import PyMouse
global m
m=PyMouse()
global cap
global y_conv
global keep_prob
global sess
global x
global ppp
from random_point_generator_image import prep
import pickle
import math



def change_shape(x):
    t=[]
    for i in x:
        t.append(i)
    t=np.array(t)
    return(t)

def noth():
    pass



def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    temp=[]
    
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            img=image[y:y+windowSize[1],x:x+windowSize[0]]
#            print(img.shape)
            if img.shape==windowSize:      
                imgg=cv2.resize(img,(28,28))
                temp.append([x,y,imgg,image.shape,windowSize])
                #temp.append(img)
            #yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
    return temp

def getpos():
    _,frame=cap.read()
    _,frame=cap.read()
    _,frame=cap.read()
    
#    frame=frame[x1:x2,y1:y2]
    frame=cv2.flip(frame,0)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    midy,midx=frame.shape
    midy=int(midy/2)
    midx=int(midx/2)
    copy=np.copy(frame)
    copy[midy-2:midy+2,midx-2:midx+2]=0
#    cv2.imshow('midpoint',copy)
    blank=np.ones_like(frame)*0
    temp1=sliding_window(frame,10,(200,200))
    
    rois=[]
    for j in temp1:
        rois.append(j[2])
    rois=np.array(rois)
    mypred=sess.run(y_conv,feed_dict={x:rois,keep_prob:1.0})    
    mypred_max=mypred[:,0].max()
    #cv2.waitKey(0)
    for ii in range(len(temp1)):
        i=temp1[ii]
        roi=i[2]
    #    cv2.imshow('roi',roi)
        roi=np.reshape(roi,(1,28,28))
        a=mypred[ii]
    #    a=sess.run(y_conv,feed_dict={x:roi,keep_prob:1.0})
        aa=a.argmax()
    #    print(a.max())   
#        t1=cv2.getTrackbarPos('threshold','trackbars')
        if aa<1:
            if a.max()==mypred_max:
                blank[i[1]:i[1]+28,i[0]:i[0]+28]=roi
                Y=i[1]
                X=i[0]
#                points.append(conv_image2real(Y+14,X+14,x2=frame.shape[0],y2=frame.shape[1]))
#                thetas.append(template_angles[aa])
        #        print(a.max())
                cv2.rectangle(frame,(i[0],i[1]),(i[0]+150,i[1]+150),0,2)
                pnt=[i[0]+100,i[1]+100]
                print(pnt)
                k=cv2.waitKey(1)
#                if k==ord('p'):
#                    finalpoints.append(pnt)
#                    print(str(ppp)+' point is collected')
#                    ppp=ppp+1
    return(pnt)

def release():
    cap.release()
    cv2.destroyAllWindows()

def test():
    ppp=0
    try:
        while(1):
            points=[]
            thetas=[]
            _,frame=cap.read()
            _,frame=cap.read()
            _,frame=cap.read()
            
        #    frame=frame[x1:x2,y1:y2]
            frame=cv2.flip(frame,0)
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            midy,midx=frame.shape
            midy=int(midy/2)
            midx=int(midx/2)
            copy=np.copy(frame)
            copy[midy-2:midy+2,midx-2:midx+2]=0
            cv2.imshow('midpoint',copy)
            blank=np.ones_like(frame)*0
            temp1=sliding_window(frame,10,(200,200))
            
            rois=[]
            for j in temp1:
                rois.append(j[2])
            rois=np.array(rois)
            mypred=sess.run(y_conv,feed_dict={x:rois,keep_prob:1.0})    
            mypred_max=mypred[:,0].max()
            #cv2.waitKey(0)
            for ii in range(len(temp1)):
                i=temp1[ii]
                roi=i[2]
            #    cv2.imshow('roi',roi)
                roi=np.reshape(roi,(1,28,28))
                a=mypred[ii]
            #    a=sess.run(y_conv,feed_dict={x:roi,keep_prob:1.0})
                aa=a.argmax()
            #    print(a.max())   
                t1=cv2.getTrackbarPos('threshold','trackbars')
                if aa<1:
                    if a.max()==mypred_max:
                        blank[i[1]:i[1]+28,i[0]:i[0]+28]=roi
                        Y=i[1]
                        X=i[0]
        #                points.append(conv_image2real(Y+14,X+14,x2=frame.shape[0],y2=frame.shape[1]))
        #                thetas.append(template_angles[aa])
                #        print(a.max())
                        cv2.rectangle(frame,(i[0],i[1]),(i[0]+150,i[1]+150),0,2)
                        pnt=[i[0]+100,i[1]+100]
                        print(pnt)
                        k=cv2.waitKey(1)
                        if k==ord('p'):
                            finalpoints.append(pnt)
                            print(str(ppp)+' point is collected')
                            ppp=ppp+1
            if ppp==5:
                break
        #                frame[i[1]:i[1]+5,i[0]:i[0]+5]=0
            #        myimg=template_images[aa]
            #        copy=np.copy(test1)
            #    copy[i[1]:i[1]+3,i[0]:i[0]+3]=0
            #    cv2.imshow('copy',copy)
            cv2.imshow('frame',frame)
        #    cv2.imshow('original',frame)
            cv2.imshow('blank',blank)
            k=cv2.waitKey(1)
        #    print("--- %s frames per second ---" % (1/(time.time() - start_time)))
#            start_time = time.time()
        
            if k==27:
                cv2.destroyAllWindows()
    #            cap.release()
                break
    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        cap.release()    
    #    time.sleep(.1)
    #    print('done')    


def calib_point():
    print('calibtrating now...')
    time.sleep(2)
    img=np.zeros([800,1530])
    eye_midpnts=[]
    wind_name='vxxx'
    cv2.namedWindow(wind_name,cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(wind_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    for i in range(10):
        img=np.zeros([800,1530])

        xx=np.random.randint(800/2-55,800/2+55)
        yy=np.random.randint(1530/2-55,1530/2+55)
        
        img[xx-3:xx+3,yy-3:yy+3]=255
        cv2.imshow(wind_name,img)
        eye_midpnts.append(getpos())
        cv2.waitKey(0)
        
        
    cv2.destroyAllWindows()
    eye=np.array(eye_midpnts)
    return eye.mean(0)


def dataset_prep(calibpoint,param):
    mydata=[]
    wind_name='DATASET prep for calib point='+str(calibpoint)
    cv2.namedWindow(wind_name,cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(wind_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    if param==0:
        vec=1
    elif param==1:
        vec=2
    elif param==2:
        vec=3
#    vec=np.array(vec)
    for i in range(20):
        sample=[]
        img,point=prep()
        cv2.imshow(wind_name,img)
        k=cv2.waitKey(0)
        [x1,y1]=point
        time.sleep(0)
        sample.append(calibpoint[0])
        sample.append(calibpoint[1])
        sample.append(x1)
        sample.append(y1)
        x=np.random.randint(0,100)
        y=np.random.randint(0,100)
#        x,y=getpos()######################################
        sample.append(x)
        sample.append(y)
#        sample=np.array(sample)
        mydata.append([sample,vec])
        if k==ord('q'):
            break
    cv2.destroyAllWindows()
    return mydata
        
def dataset_prep_calib(param):
    mydata=[]
    ite=5
    for i in range(ite):
        calibpoint=calib_point() ####################################
#        calibpoint=345,643
        mydata.append(dataset_prep(calibpoint,param)[:])
    mydata=np.array(mydata)
    mydata=np.reshape(mydata,(ite*20,2))
    mydata=mydata.tolist()
    return mydata

def dataset_prep_calib_param():
    mydata=[]
    ite=5
#    calibpoint=calib_point() ###############################
#    calibpoint=324,432
    mydata.append(dataset_prep_calib(0))
    time.sleep(5)
    print("true positive done")
    mydata.append(dataset_prep_calib(1))
    time.sleep(5)
    print("false positive done")
    mydata.append(dataset_prep_calib(2))
    time.sleep(5)
    print("do nothing done")
    mydata=np.array(mydata)
    mydata=np.reshape(mydata,(ite*20*3,2))   
    return mydata

#==============================================================================
# def visiontest():
#     filename = 'E:\\Babu\\CARL ZIESS\\hackahon_ML_new.sav'
#     model = pickle.load(open(filename, 'rb'))
#     print('svm  model loaded')
#     wind_name='vxxx'
#     cv2.namedWindow(wind_name,cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty(wind_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
#     blue=np.zeros((500,500,3))
#     green=np.zeros((500,500,3))
#     red=np.zeros((500,500,3))
#     blue[:,:,0]=255
#     green[:,:,1]=255
#     red[:,:,2]=255
#     mydict={1:green,2:blue,3:red}
#     eye_midpoint=calib_point()
# 
#     while(1):
#         print('vdd')
#         img,point=prep()
#         print('scx')
#         print('asass\n\n')
#         cv2.imshow(wind_name,img)
#         k=cv2.waitKey(0)
#         x,y=getpos()
#         
#         X=[eye_midpoint[0],eye_midpoint[1],point[0],point[1],x,y]
#         label=model.predict(X)[0]
#         result=mydict[label]
#         cv2.imshow(wind_name,result)
#         k=cv2.waitKey(0)
#         if k==ord('q'):
#             cv2.destroyAllWindows()
#             break
#             
#==============================================================================
        

def visiontest():

    wind_name='vxxx'
    cv2.namedWindow(wind_name,cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(wind_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    green=np.zeros((800,1530,3))
    red=np.zeros((800,1530,3))
    green[:,:,1]=255
    red[:,:,2]=255
    mydict={1:green,2:red}
    eye=calib_point()
#    point=[400,765]
    img,point=prep()
    count=10
    for i in range(10):
        img,point=prep()
        while(not((point[0]<325 or point[0]>475) and (point[1]<690 or point[1]>840))):            
            img,point=prep()
#            print('in while loop')
#        print('out of while loop')
        img[400-3:400+3,765-3:765+3]=255
        cv2.imshow(wind_name,img)
        k=cv2.waitKey(0)
        x,y=getpos()
        diff=math.sqrt((x-eye[0])**2+(y-eye[1])**2)
#        print(diff)
        if diff<10:
            print('!!!!!!!!!!!!!!!!!')
            cv2.imshow(wind_name,red)
            cv2.waitKey(1)
            time.sleep(.4)
            count=count-1
#        result=mydict[label]
#        cv2.imshow(wind_name,result)
#        k=cv2.waitKey(0)
        if k==ord('q'):
            cv2.destroyAllWindows()
            break
    print("Your VISION score is : "+str(count)+'!!!!!!!!!!!!!')
    cv2.destroyAllWindows()
           
def difff(xx,yy):
    XX,YY=getpos()
    diff=math.sqrt((xx-XX)**2+(yy-YY)**2)
    return diff

       
def attention():
    wind_name='vxxx'
    cv2.namedWindow(wind_name,cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(wind_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)    
    green=np.zeros((800,1530,3))
    red=np.zeros((800,1530,3))
    green[:,:,1]=255
    red[:,:,2]=255
    mydict={1:green,2:red}
    eye=calib_point()
    img,point=prep()
    tim=0
    for i in range(2):
        img,point=prep()
        cv2.imshow(wind_name,np.ones_like(img)*255)
        cv2.setWindowProperty(wind_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN) 
        cv2.waitKey(0)
        cv2.imshow(wind_name,img)
        cv2.waitKey(0)
        xx,yy=getpos()
        xx,yy=getpos()
        xx,yy=getpos()
        xx,yy=getpos()
        t1=time.time()
        d=difff(xx,yy)
        t2=time.time()-t1
        while(d<15 and t2<5):
            t2=time.time()-t1
            d=difff(xx,yy)
            pass
        cv2.destroyAllWindows()
        if d>15:
#            print('d is '+str(d))
            cv2.imshow(wind_name,red)
            cv2.waitKey(1)
            time.sleep(.5)
            
        elif t2>5:
            cv2.imshow(wind_name,green)
            cv2.waitKey(1)
            time.sleep(.5)
        tim+=t2    
    print("attention span on a scale of one to ten is :" +str(tim))
    cv2.destroyAllWindows()
            
            
def main():
    
    cap=cv2.VideoCapture(1)
    cap.release()
    time.sleep(.3)
    cap=cv2.VideoCapture(1)
    cap.release()
    cap=cv2.VideoCapture(1)
#    k=0
#    cv2.namedWindow('trackbars')
#    cv2.createTrackbar('threshold','trackbars',10,200,noth)
#    shape=[950,1920]
#    start_time = time.time()
#    black=np.zeros(shape)
#    cv2.imshow('black',black)
    global finalpoints
    finalpoints=[]
#    ppp=1
    tf.reset_default_graph()
#    ppp=0
    
    sess=tf.Session()
    saver = tf.train.import_meta_graph('E:\\Babu\\CARL ZIESS\\TEAM 007\\test3.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    graph=tf.get_default_graph()
    y_conv=graph.get_tensor_by_name('add_4:0')
    keep_prob=graph.get_tensor_by_name('Placeholder_2:0')
    x=graph.get_tensor_by_name('Placeholder:0')
    
    
    test()
    
    print('getting pos')
    getpos()
    getpos()
    
#    eye_midpoint=calib_point()

#    mydat=dataset_prep_calib_param()



    visiontest()
    attention()


if __name__=="__main__":
    main()

