import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Set page config with wide layout
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional medical UI - Light Theme
st.markdown("""
<style>
    /* Force light theme override */
    .stApp {
        background: #f8f9fa;
        color: #1a1a1a;
    }
    
    /* Override all text colors to dark */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #1a1a1a !important;
    }
    
    /* Main container styling */
    .main {
        background: #f8f9fa !important;
        color: #1a1a1a !important;
    }
    
    /* Force file uploader to light theme */
    [data-testid="stFileUploader"] {
        background: white !important;
        color: #1a1a1a !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #1a1a1a !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: white !important;
    }
    
    /* Override markdown text */
    .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Header styling */
    .header-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #0066cc;
    }
    
    .header-title {
        color: #1a1a1a;
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: #4a5568;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Diagnosis result styling */
    .diagnosis-positive {
        background: #fff5f5;
        color: #c53030;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        border: 2px solid #fc8181;
        margin: 1.5rem 0;
    }
    
    .diagnosis-negative {
        background: #f0fdf4;
        color: #15803d;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        border: 2px solid #86efac;
        margin: 1.5rem 0;
    }
    
    .confidence-text {
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff !important;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #1a1a1a !important;
    }
    
    .info-box * {
        color: #1a1a1a !important;
    }
    
    .warning-box {
        background: #fef3c7 !important;
        border-left: 4px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #1a1a1a !important;
    }
    
    .warning-box * {
        color: #1a1a1a !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: #0066cc;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background: #0052a3;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
    }
    
    /* Upload area - removed custom section, rely on native uploader */
    [data-testid="stFileUploader"] {
        background: white !important;
        padding: 2rem;
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        color: #1a1a1a !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0066cc;
    }
    
    [data-testid="stFileUploader"] * {
        color: #1a1a1a !important;
    }
    
    /* Image container */
    .image-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    
    .image-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        background: white;
        border-radius: 8px;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Metrics styling */
    .metric-container {
        background: white !important;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        color: #1a1a1a !important;
    }
    
    .metric-container * {
        color: inherit !important;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 600;
        color: #0066cc !important;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #6b7280 !important;
        margin-top: 0.5rem;
    }
    
    /* Spinner override */
    .stSpinner > div {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 1. Improved Helper Functions (Grad-CAM & Preprocessing)
# ------------------------------------------------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap for a given image.
    Adapted to handle nested VGG models.
    """
    # Access the nested VGG16 base model (first layer in Sequential)
    vgg_model = model.layers[0]
    last_conv_layer = vgg_model.get_layer(last_conv_layer_name)
    
    # Create a model from VGG input to last conv layer output
    conv_model = tf.keras.models.Model(
        inputs=vgg_model.input,
        outputs=last_conv_layer.output
    )
    
    # Create a model from last conv layer to final output
    # We need the shape excluding the batch size
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    # Apply remaining VGG layers after the last conv layer
    found_last_conv = False
    for layer in vgg_model.layers:
        if layer.name == last_conv_layer_name:
            found_last_conv = True
            continue
        if found_last_conv:
            x = layer(x)
    
    # Apply the rest of the top-level model (Flatten, Dense, etc.)
    for layer in model.layers[1:]:
        x = layer(x)
    
    classifier_model = tf.keras.models.Model(classifier_input, x)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        # Get conv output
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        # Get predictions
        preds = classifier_model(conv_output)
        
        # Check prediction shape to determine class index
        # If binary sigmoid (1 output), we focus on that single channel
        if preds.shape[-1] == 1:
            top_class_channel = preds[:, 0]
        else:
            # If softmax (2 outputs), we focus on the Pneumonia class (usually index 1)
            top_class_channel = preds[:, 1]
    
    # Gradient of the top class with respect to conv output
    grads = tape.gradient(top_class_channel, conv_output)
    
    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv output by pooled gradients
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image using thresholding and sharpening.
    """
    # 1. Normalization Fix: Convert 0-255 image to 0-1 float for this logic
    img = img.astype(np.float32) / 255.0

    # Ensure heatmap is valid numpy array
    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy()
    
    # Ensure heatmap is 2D
    if len(heatmap.shape) != 2:
        heatmap = np.squeeze(heatmap)
    
    # Clip and normalize heatmap
    heatmap = np.clip(heatmap, 0, 1)
    
    # Sharpen heatmap to focus on peaks and reduce background noise
    heatmap = np.power(heatmap, 2)
    
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB colormap (0-1 range)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Only show heatmap where activation is significant (>25%)
    # This prevents the "blue wash" over the whole image
    threshold = 0.25
    mask = heatmap_resized > threshold
    mask_expanded = mask[..., np.newaxis]
    
    # Blend only where mask is True, otherwise keep original image
    superimposed = np.where(
        mask_expanded,
        heatmap_colored * alpha + img * (1 - alpha),
        img
    )
    
    # Clip to 0-1
    superimposed = np.clip(superimposed, 0, 1)
    
    # Convert back to 0-255 uint8 for Streamlit display
    return (superimposed * 255).astype("uint8")

def preprocess_image(image):
    """
    Matches the preprocessing steps from your notebook:
    1. Resize to 224x224
    2. Convert to Array
    3. Scale to 0-255 (if not already)
    4. Apply VGG16 preprocess_input
    """
    # Resize to model input size
    image = image.resize((224, 224))
    img_array = np.array(image)

    # Ensure 3 channels (RGB)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4: # RGBA -> RGB
        img_array = img_array[..., :3]

    # Expand dims to create batch (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # Use VGG16 specific preprocessing (Zero-centering BGR)
    preprocessed_img = tf.keras.applications.vgg16.preprocess_input(img_batch.copy())
    
    return img_array, preprocessed_img

# ------------------------------------------------------------------
# 2. Main App Interface
# ------------------------------------------------------------------

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">AI Pneumonia Detection System</h1>
    <p class="header-subtitle">Advanced Deep Learning Analysis for Chest X-Ray Diagnosis</p>
</div>
""", unsafe_allow_html=True)

# Introduction section
st.markdown("""
<div class="info-box">
    <h3 style="margin-top: 0; color: #1e40af;">How It Works</h3>
    <p style="margin-bottom: 0;">
        This AI-powered system uses a VGG16 deep learning model trained on thousands of chest X-ray images 
        to detect signs of pneumonia. Upload your X-ray image below, and the AI will analyze it and provide 
        a diagnosis with visual explanations using <strong>Enhanced Grad-CAM</strong> technology.
    </p>
</div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_pneumonia_model():
    try:
        model = tf.keras.models.load_model('pneumonia_vgg16_model.h5')
        return model
    except:
        return None

model = load_pneumonia_model()

if model is None:
    st.markdown("""
    <div class="warning-box">
        <h3 style="margin-top: 0; color: #b45309;">Model Not Found</h3>
        <p style="margin-bottom: 0;">
            Please make sure <code>pneumonia_vgg16_model.h5</code> is in the same directory as this application.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Identify the last convolutional layer for Grad-CAM
    LAST_CONV_LAYER = 'block5_conv3' 

    st.markdown("<br>", unsafe_allow_html=True)
    
    file = st.file_uploader(
        "Upload Chest X-Ray Image",
        type=["jpg", "jpeg", "png"],
        help="Drag and drop your X-ray image here, or click to browse (JPG, JPEG, or PNG format)"
    )

    if file:
        st.markdown("<br>", unsafe_allow_html=True)
        
        image = Image.open(file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="image-container">
                <div class="image-label">Uploaded X-Ray Image</div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("Analyze Image", use_container_width=True)
        
        if analyze_button:
            with st.spinner("AI is analyzing the X-ray image..."):
                # 1. Preprocess
                original_array, preprocessed_batch = preprocess_image(image)
                
                # 2. Predict
                prediction = model.predict(preprocessed_batch, verbose=0)
                
                # Assuming Binary Classification: [0]=Normal, [1]=Pneumonia (or single sigmoid output)
                if prediction.shape[-1] == 1:
                    score = float(prediction[0][0])
                    is_pneumonia = score > 0.5
                    confidence = score if is_pneumonia else 1 - score
                else:
                    score = float(prediction[0][1])
                    is_pneumonia = np.argmax(prediction[0]) == 1
                    confidence = float(np.max(prediction[0]))

                # 3. Generate Improved Grad-CAM (only if pneumonia detected)
                overlay = None
                if is_pneumonia:
                    heatmap = make_gradcam_heatmap(preprocessed_batch, model, LAST_CONV_LAYER)
                    overlay = overlay_gradcam(original_array, heatmap, alpha=0.4)

                # 4. Display Results
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("<h2 style='text-align: center; color: #1a1a1a;'>Analysis Results</h2>", unsafe_allow_html=True)
                
                # Diagnosis result with custom styling
                if is_pneumonia:
                    st.markdown(f"""
                    <div class="diagnosis-positive">
                        <div>PNEUMONIA DETECTED</div>
                        <div class="confidence-text">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="warning-box">
                        <h4 style="margin-top: 0; color: #b45309;">Medical Recommendation</h4>
                        <p style="margin-bottom: 0;">
                            The AI has detected signs of pneumonia. Please consult with a qualified healthcare 
                            professional immediately for proper diagnosis and treatment. This AI analysis is a 
                            screening tool and should not replace professional medical evaluation.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="diagnosis-negative">
                        <div>✓ NO PNEUMONIA DETECTED</div>
                        <div class="confidence-text">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="info-box">
                        <h4 style="margin-top: 0; color: #1e40af;">Important Note</h4>
                        <p style="margin-bottom: 0;">
                            The AI analysis suggests no signs of pneumonia. However, if you have symptoms or concerns, 
                            please consult with a healthcare professional. This AI tool is designed to assist, not replace, 
                            professional medical diagnosis.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display metrics
                st.markdown("<br>", unsafe_allow_html=True)
                met_col1, met_col2, met_col3 = st.columns(3)
                
                with met_col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{confidence:.1%}</div>
                        <div class="metric-label">Confidence Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with met_col2:
                    result_text = "Pneumonia" if is_pneumonia else "Normal"
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{result_text}</div>
                        <div class="metric-label">Diagnosis</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with met_col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">VGG16</div>
                        <div class="metric-label">AI Model</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display images side by side (only show Grad-CAM for pneumonia cases)
                if is_pneumonia:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center; color: #1a1a1a;'>Visual Analysis</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div class="image-container">
                            <div class="image-label">Original X-Ray</div>
                        """, unsafe_allow_html=True)
                        st.image(image, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="image-container">
                            <div class="image-label">Enhanced Grad-CAM</div>
                        """, unsafe_allow_html=True)
                        st.image(overlay, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="info-box">
                        <h4 style="margin-top: 0; color: #1e40af;">Understanding the Heat Map</h4>
                        <p style="margin-bottom: 0;">
                            The Grad-CAM visualization shows which regions of the X-ray the AI focused on. 
                            We have <strong>filtered out low-confidence background noise</strong> to make the diagnosis clearer.
                            <span style="color: #dc2626; font-weight: bold;">Red areas</span> indicate regions of high attention 
                            contributing to the prediction.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Placeholder when no image is uploaded
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e5e7eb;">
                <h3 style="color: #6b7280;">No Image Uploaded</h3>
                <p style="color: #9ca3af;">Please upload a chest X-ray image to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p><strong>Disclaimer:</strong> This AI system is for educational and screening purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p>Developed with TensorFlow & Streamlit | VGG16 Deep Learning Model</p>
</div>
""", unsafe_allow_html=True)