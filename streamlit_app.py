import streamlit as st
import os
import torch
from data_loader import load_dataset
from model import AudioClassificationModel, preprocess_audio

# --- Page Layout & Setup ---
st.set_page_config(page_title="Cardiopulmonary Sound Diagnostic AI", layout="centered")
st.title("🫁 Cardiopulmonary Sound Signals Monitor")
st.write("Deploying Deep Learning to analyze and classify respiratory/cardiac audio.")

# --- 1. Background Dataset Verification (Cached) ---
@st.cache_resource
def initialize_backend():
    """Downloads dataset on app startup and instantiates model architectures."""
    # Automated dataset pull
    dataset_path = load_dataset()
    
    # Initialize mock/untrained model structure for deployment testing
    # In production, replace this with torch.load("your_trained_weights.pth")
    model = AudioClassificationModel(num_classes=3)
    model.eval() 
    return dataset_path, model

dataset_path, model = initialize_backend()



# --- 2. Interactive Front-End UI ---
st.write("---")
st.header("Analyze Audio Specimen")
uploaded_file = st.file_uploader("Upload an lung/heart sound file (.wav format)", type=["wav"])

if uploaded_file is not None:
    # Play the uploaded audio snippet
    st.audio(uploaded_file, format="audio/wav")
    
    # Temporary save path to handle raw bytes into librosa system pipeline
    temp_path = "temp_uploaded_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.info("🔄 Processing signal characteristics...")
    
    # Preprocess audio clip
    input_tensor = preprocess_audio(temp_path)
    
    if input_tensor is not None:
        # Run dummy forward inference passes
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            
        # Classes mock mapping
        classes = ["Normal Sound Pattern", "Wheeze Detected (Anomalous)", "Crackle Detected (Anomalous)"]
        prediction = classes[np.argmax(probabilities)]
        confidence = max(probabilities) * 100
        
        # Display Prediction Outputs
        st.subheader("Diagnostic Assessment Output")
        if "Normal" in prediction:
            st.success(f"Result: **{prediction}** ({confidence:.2f}% Confidence)")
        else:
            st.warning(f"Result: **{prediction}** ({confidence:.2f}% Confidence)")
            
        # Optional: Render probability chart breakdown
        st.bar_chart({classes[i]: float(probabilities[i]) for i in range(len(classes))})
    else:
        st.error("Could not read sound parameters. Please verify your .wav compression layout.")
        
    # Clean up file buffer
    if os.path.exists(temp_path):
        os.remove(temp_path)