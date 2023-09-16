import streamlit as st
from roboflow import Roboflow
import tempfile
from PIL import Image

# Initialize Roboflow
rf = Roboflow(api_key="ilMURe24auiQb8rx35IH")
project = rf.workspace().project("ai-dumping-detection")
model = project.version(3).model

def resize_image(image_path, base_width=500):
    img = Image.open(image_path)
    w_percent = base_width / float(img.size[0])
    h_size = int(float(img.size[1]) * float(w_percent))
    img = img.resize((base_width, h_size), Image.ANTIALIAS)
    img.save(image_path)

def get_prediction(image_path):
    model.predict(image_path, confidence=40, overlap=30).json()
    output_image_path = image_path.replace('.jpg', '_prediction.jpg')
    model.predict(image_path, confidence=40, overlap=30).save(output_image_path)
    return output_image_path

def main():
    st.title('Roboflow Inference Streamlit App')

    st.write("You can upload your own image or choose one of our sample images for testing.")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    image_to_process = None

    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(uploaded_file.read())
        image_to_process = temp_file.name
    else:
        test_images = ["test_image1.jpg", "test_image2.jpg"]
        test_image_choice = st.selectbox("Choose a sample image:", test_images)
        if st.button("Use Selected Sample Image"):
            image_to_process = test_image_choice

    if image_to_process:
        resize_image(image_to_process)
        st.image(image_to_process, caption='Selected Image.', use_column_width=True)
        st.write("Predicting...")
        prediction_path = get_prediction(image_to_process)
        resize_image(prediction_path)
        prediction_image = Image.open(prediction_path)
        st.image(prediction_image, caption='Predicted Image.', use_column_width=True)

if __name__ == "__main__":
    main()
