import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np


st.set_page_config(layout="wide")


@st.cache_resource
def load_model():
    print("-----------------model load")
    return YOLO("model/best.pt")


model = load_model()

###################################### sidebar
image_upload = st.sidebar.file_uploader("추론할 이미지를 선택하세요", type=["jpg", "jpeg", "png"])
conf = st.sidebar.slider("conf:", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
##################################### content
st.title("과일 야채 검출")
col1, col2 = st.columns(2)

if image_upload is not None:
    up_image = Image.open(image_upload)
    # up_image.save(os.path.join("up_img", image_upload.name))

    # 추론
    result = model(up_image, conf=conf)[0]

    result_img = result.plot()[:, :, ::-1]

    col1.header("원본")
    col1.image(up_image, width=500)

    col2.header("결과")

    col2.image(result_img, width=500)
else:
    st.write("추론할 이미지를 업로드 하세요")
