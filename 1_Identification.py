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
st.title('Person Identification')
st.subheader('Upload hand image')
st.sidebar.image("logo.png")
# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Function to preprocess and detect hands
def preprocess_and_detect_hands(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = (img / 255.0).astype(np.float32)  # Normalize to float32
    img = (img * 255).astype(np.uint8)  # Convert back to uint8
    results = hands.process(img)
    return img if results.multi_hand_landmarks else None

# Function to load models and class mappings
@st.cache_resource
def load_models():
    hand_identification_model_path = 'models/hand_identification_model.h5'
    hand_identification_model = load_model(hand_identification_model_path)
    hand_identification_class_names = ['person1', 'person10', 'person100', 'person101', 'person102', 'person103', 'person104', 'person105', 'person106', 'person107', 'person108', 'person109', 
    'person11', 'person110', 'person111', 'person112', 'person113', 'person114', 'person115', 'person116', 'person117', 'person118', 'person119', 
    'person12', 'person120', 'person121', 'person122', 'person123', 'person124', 'person125', 'person126', 'person127', 'person128', 'person129', 
    'person13', 'person130', 'person131', 'person132', 'person133', 'person134', 'person135', 'person136', 'person137', 'person138', 'person139', 
    'person14', 'person140', 'person141', 'person142', 'person143', 'person144', 'person145', 'person146', 'person147', 'person148', 'person149', 
    'person15', 'person150', 'person151', 'person152', 'person153', 'person154', 'person155', 'person156', 'person157', 'person158', 'person159', 
    'person16', 'person160', 'person161', 'person162', 'person163', 'person164', 'person165', 'person166', 'person167', 'person168', 'person169', 
    'person17', 'person170', 'person171', 'person172', 'person173', 'person174', 'person175', 'person176', 'person177', 'person178', 'person179', 
    'person18', 'person180', 'person181', 'person182', 'person183', 'person184', 'person185', 'person186', 'person187', 'person188', 'person189', 
    'person19', 'person2', 'person20', 'person21', 'person22', 'person23', 'person24', 'person25', 'person26', 'person27', 'person28', 'person29', 
    'person3', 'person30', 'person31', 'person32', 'person33', 'person34', 'person35', 'person36', 'person37', 'person38', 'person39', 'person4', 
    'person40', 'person41', 'person42', 'person43', 'person44', 'person45', 'person46', 'person47', 'person48', 'person49', 'person5', 'person50', 
    'person51', 'person52', 'person53', 'person54', 'person55', 'person56', 'person57', 'person58', 'person59', 'person6', 'person60', 'person61', 
    'person62', 'person63', 'person64', 'person65', 'person66', 'person67', 'person68', 'person69', 'person7', 'person70', 'person71', 'person72', 
    'person73', 'person74', 'person75', 'person76', 'person77', 'person78', 'person79', 'person8', 'person80', 'person81', 'person82', 'person83', 
    'person84', 'person85', 'person86', 'person87', 'person88', 'person89', 'person9', 'person90', 'person91', 'person92', 'person93', 'person94', 
    'person95', 'person96', 'person97', 'person98', 'person99']  # Example class names
    return hand_identification_model, hand_identification_class_names

# Function to cache loaded models
def get_cached_models():
    session_state = st.session_state
    if 'hand_identification_model' not in session_state:
        session_state.hand_identification_model, session_state.hand_identification_class_names = load_models()
    return session_state.hand_identification_model, session_state.hand_identification_class_names

# Main function for Streamlit app
def main():
    # Load models
    hand_identification_model, hand_identification_class_names = get_cached_models()

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

            # Example: Replace with your hand identification model prediction logic
            hand_identification_prediction = hand_identification_model.predict(np.expand_dims(hand_image, axis=0))
            predicted_class_index = np.argmax(hand_identification_prediction)
            hand_identification_predicted_class = hand_identification_class_names[predicted_class_index]

            st.subheader(f"The hand belongs to: {hand_identification_predicted_class}")
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
            captured_image = np.array(captured_image)

            # Process captured image for hand detection
            hand_image = preprocess_and_detect_hands(captured_image)

            if hand_image is not None:
                # Display captured hand image
                st.image(hand_image, caption='Detected Hand Image.', use_column_width=True)

                # Predict person from hand image
                hand_identification_prediction = hand_identification_model.predict(np.expand_dims(hand_image, axis=0))
                predicted_class_index = np.argmax(hand_identification_prediction)
                hand_identification_predicted_class = hand_identification_class_names[predicted_class_index]

                st.subheader(f"The hand belongs to: {hand_identification_predicted_class}")
                st.info('''Predictions may vary due to factors like geographic region, lighting conditions, and differences in input images, especially with webcam captures.
                     The system achieves up to 98% accuracy based on its trained dataset under ideal conditions.''')
            else:
                st.error("No hand detected in the captured image.")

if __name__ == "__main__":
    main()
