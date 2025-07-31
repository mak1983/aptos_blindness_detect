import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

# Model and class setup
MODEL_PATH = "Resnet50_aptos.keras"
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
IMG_SIZE = (320, 320)

@st.cache_resource
def load_aptos_model():
    return load_model(MODEL_PATH)

def preprocess_image(img: Image.Image):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), np.array(img)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(orig_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, orig_img.shape[1::-1])
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(orig_img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img

# Streamlit UI
st.title("APTOS Blindness Detection 👁️")
st.markdown("Upload a retinal image to predict diabetic retinopathy stage with Grad-CAM")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Load model and preprocess
    model = load_aptos_model()
    input_array, orig_img = preprocess_image(image_pil)
    prediction = model.predict(input_array)
    predicted_class = np.argmax(prediction)

    # Grad-CAM
    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer = layer.name
            break

    heatmap = make_gradcam_heatmap(input_array, model, last_conv_layer)
    overlay = overlay_heatmap((orig_img * 255).astype(np.uint8), heatmap)

    st.subheader(f"🧠 Predicted Class: {predicted_class} ({CLASS_NAMES[predicted_class]})")
    st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
