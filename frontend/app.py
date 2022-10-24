import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
import os

# Set page tab display
st.set_page_config(
    page_title="Simple Image Uploader",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# url = os.getenv('API_URL')

url = "http://localhost:8000"


# App title and description
st.header("Captioning my friends! 📸")
st.markdown(
    """
            > This is a simple image uploader app that uses a custom model to caption my friends in images.

            > Currently it is only trained on 3 people, but I plan to expand it to more people in the future.

            > I am continiously developing this app to learn more about machine learning and deployment.
            """
)

st.markdown("---")

### Create a native Streamlit file upload input
st.markdown("### Let's see if it captions my friends correctly. 👇")
img_file_buffer = st.file_uploader("Upload an image")

if img_file_buffer is not None:

    col1, col2 = st.columns(2)

    with col1:
        ### Display the image user uploaded
        st.image(
            Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️"
        )

    with col2:
        with st.spinner("Wait for it..."):
            ### Get bytes from the file buffer
            img_bytes = img_file_buffer.getvalue()

            ### Make request to  API (stream=True to stream response as bytes)
            res = requests.post(url + "/upload_image", files={"img": img_bytes})

            if res.status_code == 200:
                ### Display the image returned by the API
                st.image(res.content, caption="Image returned from API ☝️")
            else:
                st.markdown("**Oops**, something went wrong 😓 Please try again.")
                print(res.status_code, res.content)
