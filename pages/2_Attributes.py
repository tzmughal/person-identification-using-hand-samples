import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import mediapipe as mp

# Set page configuration
st.set_page_config(
    page_title="Person Identification and Attributes",
    page_icon="logo.png",
)

# Title and header
st.title('Attributes Identification')
st.subheader('Upload hand image')
st.sidebar.image("logo.png")

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)


def preprocess_and_detect_hands(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to BGR format
    img = cv2.resize(img, (224, 224))
    img = (img / 255.0).astype(np.float32)  # Normalize to float32
    img = (img * 255).astype(np.uint8)  # Convert back to uint8
    results = hands.process(img)
    return img if results.multi_hand_landmarks else None


# Function to load models and class mappings
def load_models():
    # Load accessories detection model
    accessories_detection_model_path = 'models/accessories_detection_model.h5'
    accessories_detection_model = load_model(accessories_detection_model_path)
    accessories_detection_class_names = {0: 'No Accessory', 1: 'Hand has accessory'}  # Replace with actual class names

    # Load age identification model
    age_identification_model_path = 'models/age_identification_model.h5'
    age_identification_model = load_model(age_identification_model_path)
    age_identification_class_names = {0: '18-22', 1: '23-27', 2: '28-32', 3: '33-37', 4: '38+'}  # Replace with actual class names

    # Load aspect of hand identification model
    aspect_of_hand_model_path = 'models/aspect_of_hand_identification_model.h5'
    aspect_of_hand_model = load_model(aspect_of_hand_model_path)
    aspect_of_hand_class_names = {0: 'dorsal left', 1: 'dorsal right', 2: 'palmar left', 3: 'palmar right'}  # Replace with actual class names

    # Load gender identification model
    gender_identification_model_path = 'models/gender_identification_model.h5'
    gender_identification_model = load_model(gender_identification_model_path)
    gender_identification_class_names = ['male', 'female']  # Replace with actual class names

    # Load nail polish detection model
    nail_polish_detection_model_path = 'models/nail_polish_detection_model.h5'
    nail_polish_detection_model = load_model(nail_polish_detection_model_path)
    nail_polish_detection_class_names = {0: 'No Nail Polish', 1: 'Nail Polish'}  # Replace with actual class names

    # Load skin color detection model
    skin_color_detection_model_path = 'models/skin_color_detection_model.h5'
    skin_color_detection_model = load_model(skin_color_detection_model_path)
    skin_color_detection_class_names = {0: 'dark', 1: 'fair', 2: 'medium', 3: 'very fair'}  # Replace with actual class names

    return (
        accessories_detection_model, accessories_detection_class_names,
        age_identification_model, age_identification_class_names,
        aspect_of_hand_model, aspect_of_hand_class_names,
        gender_identification_model, gender_identification_class_names,
        nail_polish_detection_model, nail_polish_detection_class_names,
        skin_color_detection_model, skin_color_detection_class_names)


def get_cached_models():
    session_state = st.session_state
    if 'models_loaded' not in session_state:
        (
            session_state.accessories_detection_model, session_state.accessories_detection_class_names,
            session_state.age_identification_model, session_state.age_identification_class_names,
            session_state.aspect_of_hand_model, session_state.aspect_of_hand_class_names,
            session_state.gender_identification_model, session_state.gender_identification_class_names,
            session_state.nail_polish_detection_model, session_state.nail_polish_detection_class_names,
            session_state.skin_color_detection_model, session_state.skin_color_detection_class_names) = load_models()
        session_state.models_loaded = True
    return (
        session_state.accessories_detection_model, session_state.accessories_detection_class_names,
        session_state.age_identification_model, session_state.age_identification_class_names,
        session_state.aspect_of_hand_model, session_state.aspect_of_hand_class_names,
        session_state.gender_identification_model, session_state.gender_identification_class_names,
        session_state.nail_polish_detection_model, session_state.nail_polish_detection_class_names,
        session_state.skin_color_detection_model, session_state.skin_color_detection_class_names)


def main():
    # Load models
    (
        accessories_detection_model, accessories_detection_class_names,
        age_identification_model, age_identification_class_names,
        aspect_of_hand_model, aspect_of_hand_class_names,
        gender_identification_model, gender_identification_class_names,
        nail_polish_detection_model, nail_polish_detection_class_names,
        skin_color_detection_model, skin_color_detection_class_names) = get_cached_models()

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Convert PIL image to OpenCV format for processing
        opencv_image = np.array(image)
        opencv_image = opencv_image[:, :, ::-1].copy()  # Convert RGB to BGR

        # Process image to detect hands
        hand_image = preprocess_and_detect_hands(opencv_image)

        if hand_image is not None:
            # Display processed hand image
            st.image(hand_image, caption='Detected Hand Image.', use_column_width=True)

            # Example using accessories detection model
            accessories_detection_prediction = accessories_detection_model.predict(np.expand_dims(hand_image, axis=0))
            predicted_accessories_class_index = np.argmax(accessories_detection_prediction)
            predicted_accessories_class = accessories_detection_class_names[predicted_accessories_class_index]

            # Example using age identification model
            age_identification_prediction = age_identification_model.predict(np.expand_dims(hand_image, axis=0))
            predicted_age_class_index = np.argmax(age_identification_prediction)
            predicted_age_class = age_identification_class_names[predicted_age_class_index]

            # Example using aspect of hand identification model
            aspect_of_hand_prediction = aspect_of_hand_model.predict(np.expand_dims(hand_image, axis=0))
            predicted_aspect_class_index = np.argmax(aspect_of_hand_prediction)
            predicted_aspect_class = aspect_of_hand_class_names[predicted_aspect_class_index]

            # Example using gender identification model
            gender_identification_prediction = gender_identification_model.predict(np.expand_dims(hand_image, axis=0))
            predicted_gender_class_index = np.argmax(gender_identification_prediction)
            predicted_gender_class = gender_identification_class_names[predicted_gender_class_index]

            # Example using nail polish detection model
            nail_polish_detection_prediction = nail_polish_detection_model.predict(np.expand_dims(hand_image, axis=0))
            predicted_nail_polish_class_index = np.argmax(nail_polish_detection_prediction)
            predicted_nail_polish_class = nail_polish_detection_class_names[predicted_nail_polish_class_index]

            # Example using skin color detection model
            skin_color_detection_prediction = skin_color_detection_model.predict(np.expand_dims(hand_image, axis=0))
            predicted_skin_color_class_index = np.argmax(skin_color_detection_prediction)
            predicted_skin_color_class = skin_color_detection_class_names[predicted_skin_color_class_index]

            # Display results
            st.subheader(f"Accessories Detection Prediction: {predicted_accessories_class}")
            st.subheader(f"Age Identification Prediction: {predicted_age_class}")
            st.subheader(f"Aspect of Hand Identification Prediction: {predicted_aspect_class}")
            st.subheader(f"Gender Identification Prediction: {predicted_gender_class}")
            st.subheader(f"Nail Polish Detection Prediction: {predicted_nail_polish_class}")
            st.subheader(f"Skin Color Detection Prediction: {predicted_skin_color_class}")

            st.info('''Predictions may vary due to factors like geographic region, lighting conditions, and differences in input images, especially with webcam captures.
                     The system achieves up to 98% accuracy based on its trained dataset under ideal conditions.''')

        else:
            st.error("No hand detected in the uploaded image.")

    # OR use camera to capture image
    st.markdown('**OR**')
    st.subheader("Use Camera to Capture Image")

    if 'camera_opened' not in st.session_state:
        st.session_state['camera_opened'] = False

    if st.session_state['camera_opened']:
        if st.button('Close Camera'):
            st.session_state['camera_opened'] = False
            st.experimental_rerun()
    else:
        if st.button('Open Camera'):
            st.session_state['camera_opened'] = True
            st.experimental_rerun()

    if st.session_state['camera_opened']:
        img_file_buffer = st.camera_input("Capture Image")

        if img_file_buffer is not None:
            captured_image = Image.open(img_file_buffer)
            st.image(captured_image, caption='Captured Image.', use_column_width=True)

            # Convert PIL image to OpenCV format for processing
            opencv_image = np.array(captured_image)
            opencv_image = opencv_image[:, :, ::-1].copy()  # Convert RGB to BGR

            # Process image to detect hands
            hand_image = preprocess_and_detect_hands(opencv_image)

            if hand_image is not None:

                # Example using accessories detection model
                accessories_detection_prediction = accessories_detection_model.predict(np.expand_dims(hand_image, axis=0))
                predicted_accessories_class_index = np.argmax(accessories_detection_prediction)
                predicted_accessories_class = accessories_detection_class_names[predicted_accessories_class_index]

                # Example using age identification model
                age_identification_prediction = age_identification_model.predict(np.expand_dims(hand_image, axis=0))
                predicted_age_class_index = np.argmax(age_identification_prediction)
                predicted_age_class = age_identification_class_names[predicted_age_class_index]

                # Example using aspect of hand identification model
                aspect_of_hand_prediction = aspect_of_hand_model.predict(np.expand_dims(hand_image, axis=0))
                predicted_aspect_class_index = np.argmax(aspect_of_hand_prediction)
                predicted_aspect_class = aspect_of_hand_class_names[predicted_aspect_class_index]

                # Example using gender identification model
                gender_identification_prediction = gender_identification_model.predict(np.expand_dims(hand_image, axis=0))
                predicted_gender_class_index = np.argmax(gender_identification_prediction)
                predicted_gender_class = gender_identification_class_names[predicted_gender_class_index]

                # Example using nail polish detection model
                nail_polish_detection_prediction = nail_polish_detection_model.predict(np.expand_dims(hand_image, axis=0))
                predicted_nail_polish_class_index = np.argmax(nail_polish_detection_prediction)
                predicted_nail_polish_class = nail_polish_detection_class_names[predicted_nail_polish_class_index]

                # Example using skin color detection model
                skin_color_detection_prediction = skin_color_detection_model.predict(np.expand_dims(hand_image, axis=0))
                predicted_skin_color_class_index = np.argmax(skin_color_detection_prediction)
                predicted_skin_color_class = skin_color_detection_class_names[predicted_skin_color_class_index]

                # Display results
            
                st.subheader(f"Accessories Detection Prediction: {predicted_accessories_class}")
                st.subheader(f"Age Identification Prediction: {predicted_age_class}")
                st.subheader(f"Aspect of Hand Identification Prediction: {predicted_aspect_class}")
                st.subheader(f"Gender Identification Prediction: {predicted_gender_class}")
                st.subheader(f"Nail Polish Detection Prediction: {predicted_nail_polish_class}")
                st.subheader(f"Skin Color Detection Prediction: {predicted_skin_color_class}")

                st.info('''Predictions may vary due to factors like geographic region, lighting conditions, and differences in input images, especially with webcam captures.
                     The system achieves up to 98% accuracy based on its trained dataset under ideal conditions.''')

            else:
                st.error("No hand detected in the uploaded image.")

if __name__ == "__main__":
    main()
