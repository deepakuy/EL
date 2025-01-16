import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load both models
counting_model = YOLO(r'best.pt')  # Model for counting
impurity_model = YOLO(r'best1.pt')  # Model for impurity detection

# Streamlit UI Configuration
st.set_page_config(page_title="Rice Grading with YOLO", page_icon="🌾", layout="wide")
st.title("🌾 Rice Grading with YOLO")
st.write("Upload an image to perform rice grain grading based on detection and impurity analysis.")

# Sidebar for options
st.sidebar.title("Options")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, help="Adjust the confidence threshold for detection."
)

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Perform inference using both models
    counting_results = counting_model(image, conf=confidence_threshold)
    impurity_results = impurity_model(image, conf=confidence_threshold)

    # Process results from the counting model
    counting_class_names = counting_model.names
    counting_class_counts = {}
    for box in counting_results[0].boxes:
        class_id = int(box.cls[0])
        label = counting_class_names[class_id]
        if label in counting_class_counts:
            counting_class_counts[label] += 1
        else:
            counting_class_counts[label] = 1

    # Process results from the impurity model
    impurity_class_names = impurity_model.names
    impurity_class_counts = {}
    for box in impurity_results[0].boxes:
        class_id = int(box.cls[0])
        label = impurity_class_names[class_id]
        if label in impurity_class_counts:
            impurity_class_counts[label] += 1
        else:
            impurity_class_counts[label] = 1

    # Display results
    st.subheader("Counting Results")
    for label, count in counting_class_counts.items():
        st.write(f"**{label}**: {count}")

    st.subheader("Impurity Results")
    for label, count in impurity_class_counts.items():
        st.write(f"**{label}**: {count}")

    # Calculate grading
    total_grains = sum(counting_class_counts.values())
    impurities = impurity_class_counts.get("impurity", 0)  # Adjust the label as per your model classes
    try:
        impurity_percentage = round((impurities / total_grains) * 100, 2)
    except ZeroDivisionError:
        impurity_percentage = 0

    st.subheader("Grading")
    if total_grains == 0:
        grade = "No grains detected."
    elif impurity_percentage <= 5:
        grade = "A (Excellent Quality)"
    elif 5 < impurity_percentage <= 10:
        grade = "B (Good Quality)"
    elif 10 < impurity_percentage <= 20:
        grade = "C (Average Quality)"
    else:
        grade = "D (Poor Quality)"

    st.write(f"**Total Grains**: {total_grains}")
    st.write(f"**Impurity Percentage**: {impurity_percentage}%")
    st.write(f"**Grade**: {grade}")

    # Visualize the counting results
    counting_image_with_boxes = np.array(image)
    for box in counting_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        if confidence >= confidence_threshold:
            cv2.rectangle(counting_image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Visualize the impurity results
    impurity_image_with_boxes = np.array(image)
    for box in impurity_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        if confidence >= confidence_threshold:
            cv2.rectangle(impurity_image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Convert and display both images
    counting_image_pil = Image.fromarray(cv2.cvtColor(counting_image_with_boxes, cv2.COLOR_BGR2RGB))
    impurity_image_pil = Image.fromarray(cv2.cvtColor(impurity_image_with_boxes, cv2.COLOR_BGR2RGB))

    st.image(counting_image_pil, caption='Counting Results', use_container_width=True)
    st.image(impurity_image_pil, caption='Impurity Results', use_container_width=True)
else:
    st.write("Please upload an image file.")

# Footer
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application uses YOLO models to grade rice grains by detecting total grains and evaluating impurities. "
    "Upload an image to analyze the results and determine the quality grade."
)