import streamlit as st


# Set page configuration
st.set_page_config(
    page_title="Person Identification and Attributes",
    page_icon="logo.png",
)


st.sidebar.image("logo.png")

st.title("About")
st.markdown("""
        This application utilizes advanced Deep Learning models to analyze hand images and identify various attributes such as age, gender, skin color, and more.

        **Key Features:**
        - Hand identification based on image analysis.
        - Age and gender estimation using machine learning models.
        - Detection of accessories, nail polish, and skin color.
        - Real-time image capture and analysis using the webcam.

        Built using Python with Streamlit for the GUI, OpenCV for image processing, and TensorFlow for machine learning models. It leverages MediaPipe for hand landmark detection and analysis.

        For support or feedback, please contact us at **[tzmughalpk@gmail.com](mailto:tzmughalpk@gmail.com)**.

        **Version 1.0.0 (Released July 2024)**
    """)
