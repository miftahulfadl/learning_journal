import pickle
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image


im = Image.open('favicon.ico')
st.set_page_config(
    page_title="Pokemon Type Classifier",
    page_icon=im,
    initial_sidebar_state= "expanded",
    menu_items={
        "Report a Bug": "https://github.com/miftahulfadl",
        "About": "Simple pokedex"
    })

with open ("target.pkl",'rb') as f:
    label = pickle.load(f)

model = tf.keras.models.load_model("model.h5")
img = None

@st.cache
def load_result(img):
    res = model.predict(img)
    res = tf.keras.layers.Softmax()(res)

    return np.argmax(res, axis = 1)[0], res


banner = Image.open('pokemmon.jpg')
a, b = st.columns([1,5])
with b:
    st.title(f" Pokemon Type Predictor ")
st.write("\n")
st.write("\n")

st.image(image=banner)

st.write("\n")

st.write("Upload your Pokemon Image")
file = st.file_uploader("",type=["png", "jpg", "jpeg"])

st.write("\n")
if file is not None:
    c, d = st.columns([10,20])
    with d:
        image = Image.open(file)
        st.image(
            image,
            caption=f"This is your pokemon",
            use_column_width='auto',
        )

        img_array = tf.keras.utils.img_to_array(image)
        img = tf.image.resize(img_array, size=(256,256))
        img = tf.expand_dims(img, axis=0)[:, :, :, :3]
    e,f = st.columns([16,20])
    with f:
        button = st.button("Predict")
    if button:
        pred = load_result(img)
        st.write("\n")
        
        st.info(f'{label[pred[0]]} pokemon')

else:
    st.warning("Please Upload Your Pokemon's Image First")

st.caption("This website is beta version (still need to development)")
