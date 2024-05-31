import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import string
import time
from PIL import Image
import os 
import sys 
import inspect 
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO
import pathlib
from ultralytics import YOLO
import numpy as np
import cv2
import glob

st.set_page_config(layout="wide")

st.title("Мониторинг вторжений на территорию")

script_directory = os.path.dirname(os.path.abspath( 
  inspect.getfile(inspect.currentframe()))) 
intrusion_folder = script_directory+ os.sep+"pictures"
os.chdir(intrusion_folder)

def update_files_list():
    print('update_files_list call')
    intruder_files_list=[]
    intruder_files_list = filter(os.path.isfile, os.listdir(intrusion_folder))
    intruder_files_list = [os.path.join(intrusion_folder, f) for f in intruder_files_list] # add path to each file
    intruder_files_list.sort(key=lambda x: -os.path.getmtime(x))
    print('intruder_files_list:', intruder_files_list)
    return intruder_files_list
    
def show_intrusions(intruder_files_list, index):
    with st.empty():
        image = Image.open(intruder_files_list[index])
        st.image(image)#, use_column_width=True

intruder_max_cnt_all=10
intruder_max_cnt_for_session=5
session_period=1
user='admin'
passw='admin'
camera_ip='192.168.1.71:554/01'
conf=0.5

if "session_id" not in st.session_state:
    st.session_state.session_id = 0
if "next_session_start" not in st.session_state:
    st.session_state.next_session_start = datetime.now()
if "stream" not in st.session_state:
    st.session_state.stream = False
def toggle_streaming():
    st.session_state.stream = not st.session_state.stream 
if "stream" not in st.session_state:
    st.session_state.stream = False
def toggle_streaming():
    st.session_state.stream = not st.session_state.stream 
st.sidebar.button(
    "Стартовать сервис", disabled=st.session_state.stream, on_click=toggle_streaming
)
st.sidebar.button(
    "Остановить сервис", disabled=not st.session_state.stream, on_click=toggle_streaming
)
if st.sidebar.checkbox('Показать настройки'):
    intruder_max_cnt_all= st.sidebar.number_input('Всего фото для хранения (при превышении цикличная перезапись наиболее старых)', format='%d', value=10)
    intruder_max_cnt_for_session= st.sidebar.number_input('Количество фото за 1 вторжение', format='%d', value= 5, key='intruder_max_cnt_for_session')
    session_period= st.sidebar.number_input('Длительность вторжения, мин', format='%d', value=1, key='session_period')
    conf=st.sidebar.number_input('Предельная вероятность обнаружения', value=0.5, min_value=0.0, max_value=1.0, step=0.1, key='conf')
    camera_ip= st.sidebar.text_input('rtsp IP камеры', value=camera_ip, key= 'camera_ip')
    user= st.sidebar.text_input('пользователь', value=user, key='user')
    passw= st.sidebar.text_input('пароль', value=passw, key='passw')
        
source='rtsp://'+user+':'+passw+'@'+camera_ip
#source= "rtsp://admin:admin@178.159.41.190:5555/01"
#source= 'rtsp://admin\:admin@192.168.1.71:554/01'
#source='D:\\python_projects\\intruder-alert\\01.avi' 
uploaded_files=[]
if st.session_state.stream==False:
    intruder_files_list= update_files_list()
    selected_file=0
    selected_file= st.number_input('Выберите файл (сортировка по убыванию даты создания)', value=0, step=1, max_value=len(intruder_files_list)-1)
    show_intrusions(intruder_files_list, selected_file)
 
placeholder = st.empty() 
if st.session_state.stream:
    # Load a pretrained YOLOv8n model
    clear = lambda: os.system('cls')
    clear()
    model_path=script_directory+ os.sep+'yolov8n.pt'
    print('model_path:', model_path)
    model = YOLO(model_path) 
    print('cv2 source2:',source)
    cap = cv2.VideoCapture(source)

    intruder_cnt=0
    session_id= st.session_state.session_id
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        results= None
        if ret:
            results = model.predict(frame, classes=[0], conf=conf, verbose=False, show=True)#
        else:
            print('error video capture')
            print('cv2 source on error:',source)
            cap = cv2.VideoCapture(source)

        if results and len(results)>0:
            for i, r in enumerate(results):
                boxes_cnt= len(r.boxes)
                if boxes_cnt>0:
                    intruder_cnt= intruder_cnt+boxes_cnt
                    print('intruder_cnt=',intruder_cnt)
                    if intruder_cnt< intruder_max_cnt_for_session+1:
                        #Plot results image
                        im_bgr = r.plot() 
                        im_rgb = Image.fromarray(im_bgr[..., ::-1])
                        with placeholder.container():
                            st.image(im_rgb)
                        c = datetime.now()
                        current_time = c.strftime('%y-%m-%d_%H_%M_%S')
                        filename=intrusion_folder+os.sep+f"{current_time}.jpg"
                        print('new file:', filename)
                        r.save(filename)

        if intruder_cnt>= intruder_max_cnt_for_session and session_id==st.session_state.session_id: 
            print('число фото вторжений превышено, обновляются данные сервиса')
            intruder_files_list= update_files_list()
            selected_file=0
            #show_intrusions(intruder_files_list, 0)
            session_id=session_id+1
            st.session_state.next_session_start= datetime.now()+ timedelta(minutes=session_period)
            print('next session id:', session_id)
            print('next session start time:', st.session_state.next_session_start)
            
        if session_id!=st.session_state.session_id and datetime.now()>=st.session_state.next_session_start:
            st.session_state.session_id= session_id
            print('new session started with id:', st.session_state.session_id)
            intruder_cnt=0
            placeholder.empty()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if st.session_state.stream==False:
            break
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print('сервис остановлен')

      
