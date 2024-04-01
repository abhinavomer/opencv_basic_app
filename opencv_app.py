import cv2
import streamlit as st
import numpy as np

st.sidebar.title("Options for OpenCV App")
rad=st.sidebar.radio("Navigation Bar",['Image Face Detection','Live_Cam','Canny_Filter','Blur_Filter','Features','Face Detector','Eye Detector'])

if rad=='Image Face Detection':

    st.title("Image Face Detector")
    img=st.file_uploader("Upload Image For Face Detection")
    stop_button=st.button('Done')
    frame_placeholder=st.empty()
    if stop_button:
        img=cv2.imdecode(np.fromstring(img.read(),np.uint8),1)
        img=cv2.resize(img,(300,400))
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray_image,1.1,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
        frame_placeholder.image(img,channels='BGR')
    cv2.destroyAllWindows()

if rad=='Live_Cam':
    source=cv2.VideoCapture(0)

    st.title("Live Cam Filter")
    frame_placeholder=st.empty()
    stop_button=st.button('stop')
    while source.isOpened() and not stop_button:
        image_filter='preview'
        has_frame,frame=source.read()
        if not has_frame:
            break
        frame=cv2.flip(frame,1)
        result=frame
        frame_placeholder.image(result,channels='BGR')
    source.release()
    cv2.destroyAllWindows()
if rad=='Face Detector':
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    st.title("Face Detector Filter")
    frame_placeholder=st.empty()
    stop_button=st.button('stop')
    cap=cv2.VideoCapture(0)
    while cap.isOpened() and not stop_button:
        response,img=cap.read()
        img=cv2.flip(img,1)
        if response:
            gr_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gr_img,1.4,7)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            frame_placeholder.image(img,channels='BGR')
    cap.release()
    cv2.destroyAllWindows()

if rad=='Eye Detector':
    eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
    st.title("Eye Detector Filter")
    frame_placeholder=st.empty()
    stop_button=st.button('stop')
    cap=cv2.VideoCapture(0)
    while cap.isOpened() and not stop_button:
        response,img=cap.read()
        img=cv2.flip(img,1)
        if response:
            gr_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=eye_cascade.detectMultiScale(gr_img,1.4,7)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            frame_placeholder.image(img,channels='BGR')
    cap.release()
    cv2.destroyAllWindows()
            
if rad=='Canny_Filter':
    source=cv2.VideoCapture(0)

    st.title("Blur Filter")
    frame_placeholder=st.empty()
    stop_button=st.button('stop')
    while source.isOpened() and not stop_button:
        image_filter='preview'
        has_frame,frame=source.read()
        if not has_frame:
            break
        frame=cv2.flip(frame,1)
        result=cv2.Canny(frame,145,150)
        frame_placeholder.image(result)
    source.release()
    cv2.destroyAllWindows()
if rad=='Blur_Filter':
    source=cv2.VideoCapture(0)

    st.title("Blur Filter")
    frame_placeholder=st.empty()
    stop_button=st.button('stop')
    while source.isOpened() and not stop_button:
        image_filter='preview'
        has_frame,frame=source.read()
        if not has_frame:
            break
        frame=cv2.flip(frame,1)
        result=cv2.blur(frame,(13,13))
        frame_placeholder.image(result,channels='BGR')
    source.release()
    cv2.destroyAllWindows()
if rad=='Features':
    source=cv2.VideoCapture(0)

    st.title("Blur Filter")
    frame_placeholder=st.empty()
    stop_button=st.button('stop')
    while source.isOpened() and not stop_button:
        image_filter='preview'
        has_frame,frame=source.read()
        if not has_frame:
            break
        frame=cv2.flip(frame,1)
        result=frame
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners=cv2.goodFeaturesToTrack(frame_gray,maxCorners=500,qualityLevel=.2,minDistance=15,blockSize=9)
        if corners is not None:
            for x,y in np.int32(corners).reshape(-1,2):
                center=(x,y)
                radius=10
                color=(0,255,0)
                t=1
                lt=cv2.LINE_8
                cv2.circle(result,center,radius,color,t,lt)

        frame_placeholder.image(result,channels='BGR')
    source.release()
    cv2.destroyAllWindows()

