import cv2
import numpy as np




import os

os.chdir('test')


a=os.listdir()


a=np.array(a)
np.random.shuffle(a)


test_images=[]
test_labels=[]

for i in a:
    img=cv2.imread(i,0)
    img=cv2.resize(img,(28,28))
    test_images.append(img)
    print(i)
    if 'eye' in i:
        test_labels.append([1,0])
    else:
        test_labels.append([0,1])
    
os.chdir('..')
os.chdir('train')



a=os.listdir()


a=np.array(a)
np.random.shuffle(a)


train_images=[]
train_labels=[]

for i in a:
    img=cv2.imread(i,0)
    img=cv2.resize(img,(28,28))
    train_images.append(img)
    print(i)
    if 'eye' in i:
        train_labels.append([1,0])
    else:
        train_labels.append([0,1])
        
        
        
os.chdir('..')

np.save('test.npy',[test_images,test_labels])
np.save('train.npy',[train_images,train_labels])