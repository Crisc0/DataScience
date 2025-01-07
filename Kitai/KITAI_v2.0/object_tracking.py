import numpy as np
import cv2
import os
import bck_adq
from motpy import Detection, MultiObjectTracker
from csv import writer 
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort

def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#path='trial.mp4'
path='pichilemu\pichilemu6224201-2.mp4'
video_stream= cv2.VideoCapture(path)
total_frames=video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
#Create folder for the result
folderpath=path.split('.')[0]+' Dataset'
if not os.path.isdir(folderpath):
   os.makedirs(folderpath)


grayMedianFrame= bck_adq.grayFrame(path)

model_spec = {
        'order_pos': 1, 'dim_pos': 2, # position is a center in 2D space; under constant velocity model
        'order_size': 0, 'dim_size': 2, # bounding box is 2 dimensional; under constant velocity model
        'q_var_pos': 1000., # process noise
        'r_var_pos': 0.1 # measurement noise
    }
tracker = MultiObjectTracker(dt=0.1)

frameCnt=0
data=[]
try:
    while(frameCnt < total_frames-1):
        ret, frame = video_stream.read()
        #% of conversion
        frameCnt+=1
        print('Process: '+str(frameCnt/total_frames*100)+'% ',end="\r")

        # Convert current frame to grayscale
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference of current frame and the median frame
        dframe = cv2.absdiff(gframe, grayMedianFrame)

        # Gaussian
        blurred = cv2.GaussianBlur(dframe, (11, 11), 0)

        #Thresholding to binarise
        ret, tframe= cv2.threshold(blurred,0,255,
                                   cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #Identifying contours from the threshold
        (cnts, _) = cv2.findContours(tframe.copy(), 
                                     cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)


        if len(cnts) < 100:
            #For each contour draw the bounding bos
            for cnt in cnts:
                x,y,w,h = cv2.boundingRect(cnt)
                tracker.step(detections=[Detection(box=np.array([int(x-w),int(y-h),int(x+2*w),int(y+2*h)]))])
                tracks = tracker.active_tracks()
                #cv2.rectangle(frame,(int(x-w),int(y-h)),(int(x+2*w),int(y+2*h)),(0,255,0),2)
                #cv2.circle(frame, (int((x+x+w)/2),int((y+y+h)/2)),5,(0,0,255),5)
                if len(tracks)>0:
                    #cv2.putText(frame, tracks[0][0], (int(x-w),int(y-h-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    
                    #['Frame','id','xmin','xmax','ymin','ymax']
                    data.append([frameCnt,tracks[0][0],int(x-w), int(x+2*w),int(y-h),int(y+2*h)])
            #Convert frames into png files
            cv2.imwrite(os.path.join(folderpath,str(frameCnt)+'.png'),frame)


        #Create video 
        #writer.write(cv2.resize(frame, (640,480)))

    #Release video object
    #video_stream.release()
    #writer.release()
except:
    pass

