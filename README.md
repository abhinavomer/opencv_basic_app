# OpenCV Streamlit App

This Streamlit app provides a user-friendly interface for performing image processing and live video filtering using OpenCV. It offers various features accessible through the sidebar navigation:

## Home
- Introduction to the OpenCV Streamlit App.
- Overview of available image processing and live video filtering options.

## Image Filters
- Choose from different image filters:
  - **Image Face & Eye Detection**: Detect faces and eyes in uploaded images.
  - **Canny Edge Detection**: Apply Canny edge detection to uploaded images.
  - **Cartoonify**: Create a cartoon version of uploaded images.
  - **Blur Your Image**: Blur uploaded images.
  - **Image Resizer**: Resize uploaded images.
  - **QR Code Scanner**: Scan QR codes from uploaded images.

## Live Filters (Note: Due to memory limitations, live filters may not work as expected in the deployed environment.)
- Real-time video filtering options:
  - **Live Cam**: Display live camera stream.
  - **Live Face Detector**: Detect faces in real-time using the webcam.
  - **Live Eye Detector**: Detect eyes in real-time using the webcam.
  - **Canny Live Filter**: Apply Canny edge detection to live video.
  - **Blur Live Filter**: Apply blur effect to live video.
  - **Features Live**: Highlight features in live video.

## Requirements
- OpenCV (cv2)
- Streamlit
- NumPy
- av
- streamlit-webrtc

## Usage
1. Install the required dependencies.
2. Run the Streamlit app using the command `streamlit run app.py`.
3. Use the sidebar to navigate between different options.
4. Upload images for image processing or use live video filtering options.

## Acknowledgements
This app utilizes the power of OpenCV for image processing and streamlit-webrtc for real-time video filtering. Special thanks to the developers of these libraries for their contributions.
