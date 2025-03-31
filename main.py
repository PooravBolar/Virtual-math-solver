import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout='wide')
st.title('Solve Anything')

# 2:1 split ratio
col1,col2 = st.columns([2,1])
with col1:
    run = st.checkbox('Run',value=True)
    # place holder for webcam image
    FRAME_WINDOW = st.image([])
    
with col2:
    st.title("Answer")
    output_text_area = st.empty()

genai.configure(api_key="ADD_YOUR_GEMINI_API_KEY")
# The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
model = genai.GenerativeModel('gemini-1.5-flash')

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)

detector = HandDetector(staticMode=False,maxHands=1,detectionCon=0.6)

def getHandInfo(img):
    hands , img = detector.findHands(img,draw=True)
    
    if hands:
        hand = hands[0]
        bbox = hand['bbox']
        lmList = hand['lmList']
        
        fingers = detector.fingersUp(hand)

        return fingers, lmList

    else:
        return None
    


prev_pos = None


def draw(info,prev_pos,canvas):
    fingers, lmList = info
    current_pos = None
    
    if fingers == [0,1,0,0,0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        
        cv2.line(canvas,current_pos,prev_pos,(255,0,255),10) 
    
    elif fingers == [1,0,0,0,0]:
        canvas = np.zeros_like(img)
    
    return current_pos, canvas

def sendToAI(model,canvas,fingers):
    if fingers == [1,1,1,1,0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve the math problem",pil_image])
        return response.text
    
canvas = None
image_combined = None
output_text = ""

while True and run:
    success , img = cap.read()
    
    img = cv2.flip(img,flipCode=1) #horizontal flip
    
    if canvas is None:
        canvas = np.zeros_like(img)
    
    info = getHandInfo(img)
    
    if info:
        fingers , lmList = info
        prev_pos,canvas = draw(info,prev_pos,canvas)
        output_text = sendToAI(model,canvas,fingers)
        
    image_combined = cv2.addWeighted(img,0.7,canvas,0.3,0)
    
    # Display webcam
    FRAME_WINDOW.image(image_combined,channels="BGR")
    # Display output
    if output_text:
        output_text_area.markdown(f"<p style='font-size:18px'>{output_text}</p>", unsafe_allow_html=True)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
