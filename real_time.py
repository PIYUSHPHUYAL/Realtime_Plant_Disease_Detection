import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the pre-trained CNN model within a name scope
with tf.name_scope("model_load"):
    model = load_model('plant_disease_cnn.h5')

# Class names
class_names = ['Apple_Black_Rot', 'Apple_Healthy', 'Apple_Scab', 'Bell_Peppe_Healthy',
               'Bell_Pepper_Bacterial_Spot', 'Cedar_Apple_Rust', 'Cherry_Healthy',
               'Cherry_Powdery_Mildew', 'Grape_Black_Rot', 'Grape_Esca_(Black_Measles)',
               'Grape_Healthy', 'Grape_Leaf_Blight', 'Maize_Cercospora_Leaf_Spot',
               'Maize_Common_Rust', 'Maize_Healthy', 'Maize_Northern_Leaf_Blight',
               'Peach_Bacterial_Spot', 'Peach_Healthy', 'Potato_Early_Blight',
               'Potato_Healthy', 'Potato_Late_Blight', 'Strawberry_Healthy',
               'Strawberry_Leaf_Scorch', 'Tomato_Bacterial_Spot', 'Tomato_Early_Blight',
               'Tomato_Healthy', 'Tomato_Late_Blight', 'Tomato_Septoria_Leaf_Spot',
               'Tomato_Yellow_Leaf_Curl_Virus']

class PlantDiseaseDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.class_names = class_names
        self.result = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to HSV for better color-based detection (green leaves)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (assumed to be the leaf)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the region and classify
            roi = img[y:y+h, x:x+w]
            if roi.size > 0:
                roi_resized = cv2.resize(roi, (256, 256))
                roi_array = roi_resized / 255.0
                roi_array = np.expand_dims(roi_array, axis=0)
                prediction = self.model.predict(roi_array)
                predicted_class_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                predicted_class = self.class_names[predicted_class_index]
                self.result = (predicted_class, confidence)

                # Add label inside the bounding box
                label = f"{predicted_class}: {confidence:.2f}%"
                cv2.putText(img, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# Streamlit app layout
st.title("Real-Time Plant Disease Detection")
st.write("Point your webcam at a plant leaf to detect diseases in real-time.")
ctx = webrtc_streamer(
    key="plant-disease-detector",
    video_transformer_factory=PlantDiseaseDetector,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)
if ctx.video_transformer and ctx.video_transformer.result:
    predicted_class, confidence = ctx.video_transformer.result
    st.write(f"**Predicted Disease:** {predicted_class}  \n**Confidence:** {confidence:.2f}%")
else:
    st.write("Processing frame...")
st.write("**Instructions:** Ensure good lighting and position the leaf clearly in the camera frame.")