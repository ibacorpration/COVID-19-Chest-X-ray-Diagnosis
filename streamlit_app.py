import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

st.set_page_config(
    page_title="COVID-19 Chest X-ray Diagnosis",
    page_icon="🦠",
    layout="centered"
)

# =============================
# Load Model
# =============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Covid_19_VGG116.h5')

model = load_model()

class_names = ['COVID', 'Normal', 'Viral Pneumonia']

# =============================
# UI Design
# =============================

st.title("🦠 COVID-19 Chest X-ray Diagnosis")
st.markdown("### Deep Learning Model using VGG16 Fine-Tuning")
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index]

    st.markdown("---")
    st.subheader("🧠 Diagnosis Result")

    if predicted_class == "Normal":
        st.success(f"Prediction: {predicted_class}")
    elif predicted_class == "COVID":
        st.error(f"Prediction: {predicted_class}")
    else:
        st.warning(f"Prediction: {predicted_class}")

    st.write(f"Confidence: **{confidence*100:.2f}%**")

    # =============================
    # Probability Chart
    # =============================

    st.markdown("### 📊 Class Probabilities")

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": prediction
    })

    st.bar_chart(prob_df.set_index("Class"))

    st.markdown("---")
    st.caption("⚠️ This AI tool is for research and educational purposes only. Not for clinical diagnosis.")