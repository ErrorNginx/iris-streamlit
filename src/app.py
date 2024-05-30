import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.model import load_model, predict
from src.utils import img_to_base64

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.write("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h1 style="text-align: center;">Prediksi Bunga Iris</h1>
    </div>
    ## Mari mengenal jenis jenis bunga iris!
    Aplikasi web interaktif ini untuk memprediksi jenis bunga Iris berdasarkan dimensi sepal dan petal. Dataset bunga Iris, yang merupakan paling umum / sering digunakan dalam pembelajaran mesin learning, terdiri dari pengukuran dari tiga spesies Iris yang berbeda (setosa, versicolor, dan virginica).
    Cukup sesuaikan slider untuk memasukkan panjang sepal, lebar sepal, panjang petal, dan lebar petal dari bunga Iris, dan biarkan model Random Forest Classifier untuk memprediksi, 
    Nikmati aplikasi ini dan pelajari lebih lanjut tentang bunga Iris yang indah!
""", unsafe_allow_html=True)

st.info(
    """
    Follow my Medium
    [Mediumnya Mang agus](https://medium.com/@agusmahari)
    """,
    icon="👾",
)

st.success(
    """
    Follow My Linkedin
    [Mang Agus](https://www.linkedin.com/in/agus-mahari/)
    """,
    icon="🗺",
)

st.markdown(
    """
    <style>
    .cover-glow {
        width: 100%;
        height: 100%;
        padding: 3px;
        top: 10px;
        left: 4;
        position: fix;
        z-index: -1;
        border-radius: 10px;  /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

img_path = "images/sidebar_streamly_avatar.png"
img_base64 = img_to_base64(img_path)
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
    unsafe_allow_html=True,
)
st.sidebar.markdown("")

st.sidebar.header('My Apps Streamlit `version 2`')

st.sidebar.subheader('memasukkan panjang sepal, lebar sepal, panjang petal, dan lebar petal dari bunga Iris')

st.sidebar.markdown("---")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4, key='sepal_length')
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
   
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.write("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h2 style="text-align: center;">User Input Parameters</h2>
    </div>
""", unsafe_allow_html=True)

st.dataframe(
    df,
    column_config={
        "sepal_length": "Sepal Length",
        "sepal_width": "Sepal Width",
        "petal_length": "Petal Length",
        "petal_width": "Petal Width",
    },
    hide_index=True,
)

clf, iris = load_model()

df_np = df.to_numpy()
prediction = predict(clf, df_np)
prediction_probability = clf.predict_proba(df_np)

st.subheader('Jenis Jenis Bunga pada Bunga Iris')
st.write(iris.target_names)

st.subheader('Prediksi Jenis Bunga')
st.write(iris.target_names[prediction])

st.subheader('kemungkinan jenis Bunga Iris')
prob_df = pd.DataFrame(prediction_probability, columns=iris.target_names)
st.write(prob_df)

st.markdown("---")

left_info_col, right_info_col = st.columns(2)

left_info_col.markdown(
    f"""
    ### Authors
    Jangan ragu untuk menghubungi saya jika ada masalah, komentar, atau pertanyaan..

    ##### Mang Agus

    - Email:  <agusmahari@gmail.com>
    - Medium: https://medium.com/@agusmahari

    """,
    unsafe_allow_html=True,
)

right_info_col.markdown(
    """
    ### Sosial Media 

    - Medium: https://www.linkedin.com/in/agus-mahari/
     """
)

right_info_col.markdown(
    """
    ### Sedang Nyari Loker nih Gan, Infor Loker dong
    """
)

with st.sidebar:
    st.markdown("---")
    st.markdown(
        '<h6>Dibuat dengan &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp oleh <a href="https://www.linkedin.com/in/agus-mahari">@Agus_mahari</a></h6>',
        unsafe_allow_html=True,
    )
