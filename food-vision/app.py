import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
from utils import load_and_prep, get_classes

@st.cache(suppress_st_warning=True)
def predicting(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df

class_names = get_classes()

st.set_page_config(page_title="Food Vision",
                   page_icon="🍔")

#### SideBar ####

st.sidebar.title("What's Food Vision ?")
st.sidebar.write("""
FoodVision is an end-to-end **CNN Image Classification Model** which identifies the food in your image. 

It can identify over 100 different food classes

It is based upom a pre-trained Image Classification Model that comes with Keras and then retrained on the infamous **Food101 Dataset**.

**Accuracy :** **`85%`**

**Model :** **`EfficientNetB1`**

**Dataset :** **`Food101`**
""")


#### Main Body ####

st.title("Food Vision 🍔📷")
st.header("Identify what's in your food photos!")
st.write("To know more about this app, visit [**GitHub**](https://github.com/gauravreddy08/food-vision)")
file = st.file_uploader(label="Upload an image of food.",
                        type=["jpg", "jpeg", "png"])


model = tf.keras.models.load_model("./models/EfficientNetB1.hdf5")


st.sidebar.markdown("Created by **Gaurav Reddy**")
st.sidebar.markdown(body="""

<th style="border:None"><a href="https://twitter.com/gaurxvreddy" target="blank"><img align="center" src="https://bit.ly/3wK17I6" alt="gaurxvreddy" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://linkedin.com/in/gauravreddy08" target="blank"><img align="center" src="https://bit.ly/3wCl82U" alt="gauravreddy08" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://github.com/gauravreddy08" target="blank"><img align="center" src="https://bit.ly/3c2onZS" alt="16034820" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://instagram.com/gaurxv_reddy" target="blank"><img align="center" src="https://bit.ly/3oZABHZ" alt="gaurxv_reddy" height="40" width="40" /></a></th>

""", unsafe_allow_html=True)

if not file:
    st.warning("Please upload an image")
    st.stop()

else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class, pred_conf, df = predicting(image, model)
    st.success(f'Prediction : {pred_class} \nConfidence : {pred_conf*100:.2f}%')
    st.write(alt.Chart(df).mark_bar().encode(
        x='F1 Scores',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='F1 Scores'
    ).properties(width=600, height=400))
