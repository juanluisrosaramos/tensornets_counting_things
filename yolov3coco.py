# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 19:40:49 2018

@author: Baakchsu
"""

import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time
from google.cloud import storage
from PIL import Image


def copy_images(path):
    client=storage.Client()

    # Create this folder locally
    if not os.path.exists(path):
        os.makedirs(path)
    # Retrieve all blobs with a prefix matching the folder
    bucket=client.get_bucket(path)
    blobs=list(bucket.list_blobs())
    print ('BLOBS',blobs)
    for blob in blobs:
        print ('BLOB',blob)
        if(not blob.name.endswith("/")):
            blob.download_to_filename(path+'/'+blob.name)
#create array with a list of images
    PATH_TO_TEST_IMAGES_DIR = path
    TEST_IMAGE_PATHS = []
    for file in os.listdir(PATH_TO_TEST_IMAGES_DIR):
        if file.endswith(".jpg"):
            TEST_IMAGE_PATHS = TEST_IMAGE_PATHS + [os.path.join(PATH_TO_TEST_IMAGES_DIR,file)]
    IMAGE_SIZE = (12, 8)
    return(TEST_IMAGE_PATHS)


if __name__ == "__main__":
    TEST_IMAGE_PATHS = copy_images(sys.argv[1])
    inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
    model = nets.YOLOv3COCO(inputs, nets.Darknet19)
    #model = nets.YOLOv2(inputs, nets.Darknet19)

    #frame=cv2.imread("D://pyworks//yolo//truck.jpg",1)

    classes={'0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'}
    list_of_classes=[0,1,2,3,5,7]
    with tf.Session() as sess:
        sess.run(model.pretrained())
    #"D://pyworks//yolo//videoplayback.mp4"    
        #cap = cv2.VideoCapture("G://ani_pers//videoplayback.mp4")
        for image_path in TEST_IMAGE_PATHS:
        #while(cap.isOpened()):
            frame = Image.open(image_path)
            #ret, frame = cap.read()
            img=cv2.resize(frame,(416,416))
            imge=np.array(img).reshape(-1,416,416,3)
            start_time=time.time()
            preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
        
            print("--- %s seconds ---" % (time.time() - start_time)) 
            boxes = model.get_boxes(preds, imge.shape[1:3])
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)

            cv2.resizeWindow('image', 700,700)
            #print("--- %s seconds ---" % (time.time() - start_time)) 
            boxes1=np.array(boxes)
            for j in list_of_classes:
                count =0
                if str(j) in classes:
                    lab=classes[str(j)]
                if len(boxes1) !=0:
                    
                    
                    for i in range(len(boxes1[j])):
                        box=boxes1[j][i] 
                        
                        if boxes1[j][i][4]>=.40:
                            
                                
                            count += 1    
    
                            cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
                            cv2.putText(img, lab, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), lineType=cv2.LINE_AA)
                print(lab,": ",count)
        
                
            cv2.imshow("image",img)  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break          




    cap.release()
    cv2.destroyAllWindows()    