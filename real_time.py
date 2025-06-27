import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = tf.keras.models.load_model('plant_disease_cnn.h5')

# Get class names from the model's training data (assuming same structure as in notebook)
class_names = [
    'Apple_Black_Rot', 'Apple_Healthy', 'Apple_Scab', 'Bell_Peppe_Healthy',
    'Bell_Pepper_Bacterial_Spot', 'Cedar_Apple_Rust', 'Cherry_Healthy',
    'Cherry_Powdery_Mildew', 'Grape_Black_Rot', 'Grape_Esca_(Black_Measles)',
    'Grape_Healthy', 'Grape_Leaf_Blight', 'Maize_Cercospora_Leaf_Spot',
    'Maize_Common_Rust', 'Maize_Healthy', 'Maize_Northern_Leaf_Blight',
    'Peach_Bacterial_Spot', 'Peach_Healthy', 'Potato_Early_Blight',
    'Potato_Healthy', 'Potato_Late_Blight', 'Strawberry_Healthy',
    'Strawberry_Leaf_Scorch', 'Tomato_Bacterial_Spot', 'Tomato_Early_Blight',
    'Tomato_Healthy', 'Tomato_Late_Blight', 'Tomato_Septoria_Leaf_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus'
]

def detect_leaves(frame):
    """Detect leaf-like objects in the frame using computer vision techniques"""
    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for green colors (leaves)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask for green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    leaf_boxes = []
    min_area = 1000

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:  # Minimum size filter
                leaf_boxes.append((x, y, w, h))

    return leaf_boxes

def preprocess_frame(frame):
    # Resize to 256x256 (model input size)
    img = cv2.resize(frame, (256, 256))
    # Convert to array and normalize
    img_array = image.img_to_array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_frame(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = class_names[predicted_class_idx]
    return predicted_class, confidence

# Streamlit app
st.title("Real-Time Plant Disease Detection")

# Placeholder for video feed
frame_placeholder = st.empty()

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    st.error("Error: Could not access webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from webcam.")
            break

        # Detect leaves and get bounding boxes
        leaf_boxes = detect_leaves(frame)

        # For each detected leaf, predict disease
        for x, y, w, h in leaf_boxes:
            # Extract leaf region
            leaf_roi = frame[y:y+h, x:x+w]

            # Predict disease for this leaf region
            predicted_class, confidence = predict_frame(leaf_roi)

            # Choose color based on health status
            if 'Healthy' in predicted_class:
                box_color = (0, 255, 0)  # Green for healthy
            else:
                box_color = (0, 0, 255)  # Red for diseased

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Draw label with class name and confidence
            label = f"{predicted_class} ({confidence:.2f}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert BGR (OpenCV) to RGB (Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame in Streamlit
        frame_placeholder.image(frame_rgb, channels="RGB")

    # Release the capture when done
    cap.release()