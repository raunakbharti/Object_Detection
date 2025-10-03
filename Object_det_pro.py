import streamlit as st
import cv2
import numpy as np
import requests
import os

file=open("D:/Gen AI/DL by Aditya Sir/objects_list.txt")
li=file.read().split('\n')
classes=list(map(str.strip,li))
file.close()

cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"

if os.path.exists("D:/Gen AI/DL by Aditya Sir/Dataset/yolov4.cfg") & os.path.exists("D:/Gen AI/DL by Aditya Sir/Dataset/yolov4.weights"):
    pass
else:
    resp=requests.get(cfg_url)
    file=open("D:/Gen AI/DL by Aditya Sir/Dataset/yolov4.cfg",'wb')
    file.write(resp.content)
    file.close()

    resp=requests.get(weights_url)
    file=open("D:/Gen AI/DL by Aditya Sir/Dataset/yolov4.weights",'wb')
    file.write(resp.content)
    file.close()

model=cv2.dnn_DetectionModel("D:/Gen AI/DL by Aditya Sir/Dataset/yolov4.cfg","D:/Gen AI/DL by Aditya Sir/Dataset/yolov4.weights")
model.setInputSize(416,416)
model.setInputScale(1/255)

def detect(path): 
    file_bytes = np.asarray(bytearray(path.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    classIds,classProbs,bboxes=model.detect(img,confThreshold=.75,nmsThreshold=.5)
  
    for box,cls,prob in zip(bboxes,classIds,classProbs):
        x,y,w,h=box
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classes[cls]}({prob:.2f})',(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)

    return img

st.markdown(
    """
    <style>
    /* Remove default padding and set full width */
    .block-container {
        max-width: 90% !important;
        padding-left: 0rem;
        padding-right: 0rem;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <div style='background-color: #4CAF50; color: white; padding: 10px 15px; border-radius: 10px; text-align: center;'>
        <h1 style='margin: 0;'>
           🕵️‍♂️ Object Detection Model
        </h1>
    </div>
""", unsafe_allow_html=True)


st.sidebar.image("D:/Gen AI/DL by Aditya Sir/odm_img.jpg")
with st.sidebar.expander("📞Contact us"):
    st.write("999999999")
with st.sidebar.expander("🧑‍🤝‍🧑About us"):
    st.write("🤖 We are a group of AI Engineers working on CNNs")

uploaded_file = st.file_uploader("Choose a file", type=["png","jpg","jfif"])

col1,col2=st.columns(2)
with col1:
    if uploaded_file:
        st.image(uploaded_file)
        btn=st.button("Prediction")
        with col2:
            if btn:
                img=detect(uploaded_file)
                st.image(img,channels="BGR")
              


