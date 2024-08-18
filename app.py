import librosa
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load your models
model = load_model('lstm_model.h5')
model_f = load_model('lstm_model_filter.h5')

# Function to convert audio to DataFrame
def audio_to_dataframe(filename):
    y, sr = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    df = pd.DataFrame(mfccs.T)
    return df

def predict_class(df, model):
    X = df.values
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape to match model input
    pred = model.predict(X)
    most_common_class = np.argmax(np.sum(pred, axis=0))
    return most_common_class

# Streamlit interface
st.title("🎙️ Baby Cry Identification 👶")

# List of allowed sound file extensions
allowed_extensions = ["wav", "mp3", "ogg", "flac", "m4a", "aac", "wma", "aiff", "alac", "opus"]

# Upload audio file
uploaded_file = st.file_uploader("Upload a sound file", type=allowed_extensions)

if uploaded_file is not None:
    # Display the uploaded audio file
    st.audio(uploaded_file)

    # Convert audio to DataFrame and display
    df = audio_to_dataframe(uploaded_file)

    # Make prediction and display result
    class_mapping_f = ['cry', 'silence', 'noise', 'laugh']
    predicted_class = predict_class(df, model_f)
    predict_filter = class_mapping_f[predicted_class]

    if predict_filter == "cry":
        class_mapping = ['hungry', 'tired', 'burping', 'belly_pain', 'discomfort']
        predicted_class = predict_class(df, model)
        predicted_class = class_mapping[predicted_class]

        if predicted_class == 'hungry':
            st.markdown(
                """
                <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังหิว </h3>
                </div>
                """, unsafe_allow_html=True
            )
            st.write("ลูกจะร้องไห้สั้น ๆ ดูดปากดูดมือให้ลูกกินนมทุก 2 - 3 ชั่วโมงในทารกแรกเกิด ซึ่งถ้าลูกหิวมากก็จะร้องไห้โวยวาย ดังนั้นคุณแม่อย่าปล่อยลูกหิวมากจนเกินไปนะ")
        elif predicted_class == 'tired':
            st.markdown(
                """
                <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังเหนื่อย </h3>
                </div>
                """, unsafe_allow_html=True
            )
            st.write("หยุดการเล่น และพาลูกออกจากสถาณการณ์หรือแสถานที่แห่งนั้น ปลอบโอ๋ให้หยุด")
        elif predicted_class == 'burping':
            st.markdown(
                """
                <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังจะเรอ </h3>
                </div>
                """, unsafe_allow_html=True
            )
            st.write("จับลูกเรอระหว่างกินนมและหลังกินนม ทามหาหิงค์ ถ้าเป็นคุณแม่ที่ให้นมลูกหลีกเลี่ยงอาหารที่ทำให้ลูกมีลมในท้อง เช่นประเภทถั่ว ถ้าลูกกินนมผงก็เลือกเป็นสูตรที่ย่อยง่าย")
        elif predicted_class == 'belly_pain':
            st.markdown(
                """
                <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังปวดท้อง </h3>
                </div>
                """, unsafe_allow_html=True
            )
            st.write("ถ้ามีไข้ อาเจียน ถ่ายเหลว กระสับกระส่าย กินนมน้อยลง ร้องไห้งอแง กรณีนี้พาลูกหาหมอ")
        elif predicted_class == 'discomfort':
            st.markdown(
                """
                <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังอึดอัด </h3>
                </div>
                """, unsafe_allow_html=True
            )
            st.write("สังเกตุอาการตัวร้อนไหม หรือมีความผิดปกติตามร่างกาย ให้รีบพาไปหาหมอ")
    else:
        st.markdown(
            """
            <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                <h3 style='text-align: center; color: red; font-size: 180%'> ❌ ไม่ใช่เสียงร้องของเด็ก ❌ </h3>
            </div>
            """, unsafe_allow_html=True
        )
