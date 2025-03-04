import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Load your trained PyTorch model
mlmodel = torch.load('pneumonia_detection_model.pth', map_location=torch.device('cpu'))  # Load your model
    
# Define the image preprocessing pipeline
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit app
st.title("Pneumonia Detection from X-Ray Images")
st.write("Upload a chest X-ray image to check for signs of pneumonia.")

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)

    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Make prediction
    with torch.no_grad():
        output = mlmodel(input_tensor)
        prediction = torch.sigmoid(output).item()  
    # Display the result
    threshold = 0.5  # Adjust threshold as needed
    if prediction > threshold:
        st.error(f"Prediction: **Pneumonia Detected** (Confidence: {prediction:.2f})")
    else:
        st.success(f"Prediction: **No Pneumonia Detected** (Confidence: {1 - prediction:.2f})")
        