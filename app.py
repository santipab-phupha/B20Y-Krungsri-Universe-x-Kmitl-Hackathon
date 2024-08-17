import pyaudio
import wave
import librosa
import pandas as pd
import tempfile
import time
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('lstm_model.h5')
model_f = load_model('lstm_model_filter.h5')

# JavaScript to request microphone access
st.markdown(
    """
    <script>
    async function getMicrophoneAccess() {
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            alert('Please allow microphone access in your browser settings.');
        }
    }
    getMicrophoneAccess();
    </script>
    """, unsafe_allow_html=True)

# Function to record audio with circular progress
def record_audio_with_progress(filename, duration=5, channels=1, rate=44100, chunk=1024):
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=pyaudio.paInt16, channels=channels,
                        rate=rate, input=True,
                        input_device_index=0,  # Change this index to the correct one
                        frames_per_buffer=chunk)

    frames = []

    progress_circle = st.empty()
    progress_text = st.empty()

    start_time = time.time()
    total_frames = int(rate / chunk * duration)

    for i in range(total_frames):
        data = stream.read(chunk)
        frames.append(data)
        
        # Update progress
        elapsed_time = time.time() - start_time
        progress = min(elapsed_time / duration, 1.0)
        remaining_time = max(duration - elapsed_time, 0)

        progress_circle.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; width: 150px; height: 150px; margin: 0 auto;">
                <div style="position: relative; width: 100%; height: 100%;">
                    <svg viewBox="0 0 36 36" style="position: absolute; width: 100%; height: 100%;">
                        <path d="M18 2.0845
                            a 15.9155 15.9155 0 0 1 0 31.831
                            a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke="#E0E0E0"
                            stroke-width="2"
                            stroke-dasharray="100, 100"
                        />
                        <path d="M18 2.0845
                            a 15.9155 15.9155 0 0 1 0 31.831
                            a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke="#4CAF50"
                            stroke-width="2"
                            stroke-dasharray="{progress * 100}, 100"
                        />
                    </svg>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 24px; font-weight: bold;">
                        {int(progress * 100)}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        progress_text.markdown(f"Time remaining: **{remaining_time:.1f} seconds**")

    progress_text.markdown("Recording saved successfully!")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording to a file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

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

# Function to play audio
def play_audio(filename):
    st.audio(filename)

# Streamlit interface
st.title("🎙️ Baby Cry Identification 👶")

duration = st.slider("Select duration of recording (seconds)", 1, 10, 5)

if st.button("Start Recording", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        record_audio_with_progress(temp_audio_file.name, duration)
        st.success("Recording has been completed!")
        play_audio(temp_audio_file.name)
        # Convert audio to DataFrame and display
        df = audio_to_dataframe(temp_audio_file.name)
        # Make prediction and display result
        class_mapping_f = ['cry', 'silence', 'noise', 'laugh']
        predicted_class = predict_class(df, model_f)
        predict_filter = class_mapping_f[predicted_class]
        if predict_filter == "cry":
            class_mapping = ['hungry', 'tired', 'burping', 'belly_pain', 'discomfort']
            predicted_class = class_mapping[predicted_class]
            if predicted_class == 'hungry':
                st.markdown(
                        f"""
                        <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                            <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังหิว </h3>
                            </div>
                            """,unsafe_allow_html=True
                )
                st.write("ลูกจะร้องไห้สั้น ๆ ดูดปากดูดมือให้ลูกกินนมทุก 2 - 3 ชั่วโมงในทารกแรกเกิด ซึ่งถ้าลูกหิวมากก็จะร้องไห้โวยวาย ดังนั้นคุณแม่อย่าปล่อยลูกหิวมากจนเกินไปนะ")           
            if predicted_class == 'tired':
                st.markdown(
                        f"""
                        <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                            <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังเหนื่อย </h3>
                            </div>
                            """,unsafe_allow_html=True
                )
                st.write("หยุดการเล่น และพาลูกออกจากสถาณการณ์หรือแสถานที่แห่งนั้น ปลอบโอ๋ให้หยุด")
            if predicted_class == 'burping':
                st.markdown(
                        f"""
                        <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                            <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังจะเรอ </h3>
                            </div>
                            """,unsafe_allow_html=True
                )
                st.write("จับลูกเรอระหว่างกินนมและหลังกินนม ทามหาหิงค์ ถ้าเป็นคุณแม่ที่ให้นมลูกหลีกเลี่ยงอาหารที่ทำให้ลูกมีลมในท้อง เช่นประเภทถั่ว ถ้าลูกกินนมผงก็เลือกเป็นสูตรที่ย่อยง่าย")
            if predicted_class == 'belly_pain':
                st.markdown(
                        f"""
                        <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                            <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังปวดท้อง </h3>
                            </div>
                            """,unsafe_allow_html=True
                )
                st.write("ถ้ามีไข้ อาเจียน ถ่ายเหลว กระสับกระส่าย กินนมน้อยลง ร้องไห้งอแง กรณีนี้พาลูกหาหมอ")
            if predicted_class == 'discomfort':
                st.markdown(
                        f"""
                        <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                            <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลังอึดอัด </h3>
                            </div>
                            """,unsafe_allow_html=True
                )
                st.write("สังเกตุอาการตัวร้อนไหม หรือมีความผิดปกติตามร่างกาย ให้รีบพาไปหาหมอ")
        else:
            st.markdown(
                    f"""
                    <div style='border: 2px solid red; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: red; font-size: 180%'> ❌ ไม่ใช่เสียงร้องของเด็ก ❌ </h3>
                        </div>
                        """,unsafe_allow_html=True
            )   


if st.button("auto",use_container_width=True):
    click_count = 0
    if click_count % 2 == 0:
        st.write("เริ่มจร้าาา")
        if click_count % 2 == 0:
            while True:
                if click_count % 2 == 0:
                    with st.status('ตรวจสอบ'):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                            record_audio_with_progress(temp_audio_file.name, duration)
                            st.success("Recording has been completed!")
                            play_audio(temp_audio_file.name)
                            # Convert audio to DataFrame and display
                            df = audio_to_dataframe(temp_audio_file.name)
                            # Make prediction and display result
                            predicted_class = predict_class(df, model)
                            class_mapping = ['hungry', 'tired', 'burping', 'belly_pain', 'discomfort']
                            predicted_class = class_mapping[predicted_class]
                            st.markdown(
                                    f"""
                                    <div style='border: 2px solid blue; border-radius: 5px; padding: 5px; background-color: white;'>
                                        <h3 style='text-align: center; color: green; font-size: 180%'> ตอนนี้เด็กกำลัง {predicted_class} </h3>
                                        </div>
                                        """,unsafe_allow_html=True
                            )
                else:
                    break
