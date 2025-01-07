import numpy as np
import cv2
import random

def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
def grayFrame(path):
    video_stream = cv2.VideoCapture(path)

    # Randomly select 30 frames
    frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video_stream.read()
        frames.append(frame)
    video_stream.release()
    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    # Calculate the average along the time axis
    sample_frame=frames[10]
    avgFrame = np.average(frames, axis=0).astype(dtype=np.uint8)
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    return grayMedianFrame