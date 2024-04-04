import cv2
import streamlit as st
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

op=st.sidebar.radio("Options",['Home','Image_Filters',"Live_Filters"])
if op=='Home':
    st.title("WELCOME TO OPEN CV APP")
    st.write("Select one option on sidebar to proceed:-")
if op=='Image_Filters':
    st.sidebar.title("Different Filters:-")
    mad=st.sidebar.radio("Image Filters", ['Image Face & Eye Detection','Canny Edge Detection','Cartoonify','Blur Your Image','Image Resizer','QR Code Scanner'])
    if mad== 'Image Face & Eye Detection':
        st.title("Image Face & Eye Detection")
        img = st.file_uploader("Upload Image For Face Detection")
        stop_button = st.button('Done')
        a,b=st.columns(2)
        frame_placeholder1= a.empty()
        frame_placeholder2= b.empty()
        if stop_button:
            img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
            img = cv2.resize(img, (300, 400))
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                img_face = cv2.rectangle(img.copy(),(x,y),(x+w,y+h),(255,0,0),5)
            a.text("Face")
            frame_placeholder1.image(img_face,channels='BGR')
            eyes = eye_cascade.detectMultiScale(gray,1.1,4)
            img_eyes = img.copy()  # Ensure img_eyes is defined regardless of the condition
            for (ex,ey,ew,eh) in eyes:
                img_eyes = cv2.rectangle(img_eyes,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  # Update img_eyes
            b.text('Eyes')
            frame_placeholder2.image(img_eyes,channels='BGR')
            cv2.destroyAllWindows()
    if mad== 'Canny Edge Detection':
        st.title("Edge Detection")
        img = st.file_uploader("Upload Image For Edge Detection")
        stop_button = st.button('Done')
        frame_placeholder= st.empty()
        if stop_button:
            img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
            img = cv2.resize(img, (300, 400))
            edge = cv2.Canny(img,50,150)
            frame_placeholder.image(edge)
        cv2.destroyAllWindows()
    if mad== 'Cartoonify':
        st.title("Cartoonify")
        img = st.file_uploader("Upload Image")
        stop_button = st.button('Done')
        frame_placeholder= st.empty()
        if stop_button:
            img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
            img = cv2.resize(img, (300, 400))
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray=cv2.medianBlur(gray,5)
            edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
            color=cv2.bilateralFilter(img,9,250,250)
            cartoon=cv2.bitwise_and(color,color,mask=edges)
            frame_placeholder.image(cartoon,channels='BGR')
        cv2.destroyAllWindows()
    if mad== 'Blur Your Image':
        st.title("Blur Your Image")
        img = st.file_uploader("Upload Image ")
        stop_button = st.button('Done')
        frame_placeholder= st.empty()
        if stop_button:
            img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
            img = cv2.resize(img, (300, 400))
            img = cv2.flip(img, 1)
            img = cv2.blur(img, (13,13))
            frame_placeholder.image(img,channels='BGR')
        cv2.destroyAllWindows()
    if mad== 'Image Resizer':
        st.title("Resize Your Image")
        img = st.file_uploader("Upload Image ")
        width=int(st.number_input("Enter Width"))
        height=int(st.number_input("Enter Height"))
        stop_button = st.button('Done')
        frame_placeholder= st.empty()
        if stop_button:
            img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
            img = cv2.resize(img, (width, height))
            frame_placeholder.image(img,channels='BGR')
        cv2.destroyAllWindows()
    if mad== 'QR Code Scanner':
        st.title("Scan QR Code")
        img = st.file_uploader("Upload QR Code/Image")
        stop_button = st.button('Done')
        frame_placeholder= st.empty()
        if stop_button:
            img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
            detector=cv2.QRCodeDetector()
            data,bbox,straight_qrcode=detector.detectAndDecode(img)
            st.write(data)









if op=='Live_Filters':
    st.sidebar.title("Different Filters:-")
    st.sidebar.write("Live Filters (Working fine on local machine but not working here due to memory issue.)")
    rad = st.sidebar.radio("Live Filters", ['Live_Cam', 'Canny_live_Filter', 'Blur_live_Filter', 'Features_live', 'Live_Face Detector', 'Live_Eye Detector'])

    if rad == 'Live_Cam':
        st.title("Live Cam ")
        def callback(frame):
            img = frame.to_ndarray(format="bgr24")
            img=cv2.resize(img,(100,100))
            img = cv2.flip(img, 1)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(        
            key="live_cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.1.google.com:19302"]}],
            }),
            video_frame_callback=callback,
            media_stream_constraints={"video": {"width": {"ideal": 300},"height": {"ideal": 400},"frameRate": {"ideal": 5},"audio": False}},
            async_processing=True,

            #video_html_attrs={"style":{"width":"50%"}}
        )

    if rad == 'Live_Face Detector':
        st.title("Face Detector Filter")
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        def callback(frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            gr_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gr_img, 1.4, 7)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="face_detector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.1.google.com:19302"]}],
            }),
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    if rad == 'Live_Eye Detector':
        st.title("Eye Detector Filter")
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        def callback(frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            gr_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gr_img, 1.4, 7)
            for (x, y, w, h) in eyes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="eye_detector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.1.google.com:19302"]}],
            }),
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    if rad == 'Canny_live_Filter':
        st.title("Canny Filter")
        def callback(frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="canny_filter",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.1.google.com:19302"]}],
            }),
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    if rad == 'Blur_live_Filter':
        st.title("Blur Filter")
        def callback(frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            img = cv2.blur(img, (13, 13))
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="blur_filter",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.1.google.com:19302"]}],
            }),
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    if rad == 'Features_live':
        st.title("Features Filter")
        def callback(frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            result = img
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(frame_gray, maxCorners=500, qualityLevel=.2, minDistance=15, blockSize=9)
            if corners is not None:
                for x, y in np.int32(corners).reshape(-1, 2):
                    center = (x, y)
                    radius = 10
                    color = (0, 255, 0)
                    t = 1
                    lt = cv2.LINE_8
                    cv2.circle(result, center, radius, color, t, lt)
            return av.VideoFrame.from_ndarray(result, format="bgr24")

        webrtc_streamer(
            key="features_filter",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.1.google.com:19302"]}],
            }),
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
