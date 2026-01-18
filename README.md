# Pneumonia Detection XAI

An intelligent Computer-Aided Diagnosis (CAD) system for automatic pneumonia detection from chest X-ray images using deep learning and explainable AI.

## Overview

This project implements a medical AI system that serves as a "second opinion" for healthcare professionals in diagnosing pneumonia from chest X-rays. The system combines high-accuracy classification with visual explanations through Grad-CAM heatmaps, making AI decisions transparent and trustworthy.

## Key Features

- **Automated Classification**: Binary classification (Normal/Pneumonia) using deep learning
- **Explainable AI (XAI)**: Grad-CAM visualization highlights pathological regions
- **High Sensitivity**: 99.74% recall to minimize missed diagnoses
- **Transfer Learning**: VGG16 architecture pre-trained on ImageNet for efficient learning
- **Medical-Grade Performance**: Optimized for screening and early detection

## Methodology

### Architecture

The system uses a **VGG16-based architecture** with the following components:

- **Base Model**: VGG16 convolutional layers (frozen) for feature extraction
- **Custom Classifier**:
  - Global Average Pooling layer
  - Dense layer (256 neurons, ReLU activation)
  - Dropout (0.5) for regularization
  - Output layer (1 neuron, Sigmoid activation)

### Data Processing

- **Dataset**: 5,863 chest X-ray images (Normal/Pneumonia classes)
- **Preprocessing**: Resize to 224×224, normalization
- **Data Augmentation**: Random rotation, zoom, contrast adjustment
- **Class Balancing**: Class weights to handle imbalanced data

### Explainability (XAI)

**Grad-CAM** (Gradient-weighted Class Activation Mapping) is implemented to visualize which regions of the X-ray influenced the model's decision. This transparency builds trust with medical professionals and helps validate predictions.

## Performance Metrics

| Metric | Score |
|--------|-------|
| **Recall (Sensitivity)** | 99.74% |
| **F1-Score** | 90.25% |
| **AUC** | 0.9726 |

The model is optimized for high sensitivity to avoid false negatives (missed pneumonia cases), which is critical in medical screening applications.

## Technologies

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization
- **OpenCV** - Image processing
- **Streamlit** - Web interface

## Project Structure

```
├── app.py                      # Streamlit web application
├── pneumonia_detection.ipynb   # Training notebook
└── README.md                   # This file
```

## Usage

### Training (Jupyter Notebook)

Open and run `pneumonia_detection.ipynb` to train the model from scratch or fine-tune the existing model.

### Web Application

Run the Streamlit app for interactive pneumonia detection:

```bash
streamlit run app.py
```

Upload a chest X-ray image and the system will:
1. Analyze the image using the trained model
2. Provide a diagnosis (Normal/Pneumonia)
3. Display confidence score
4. Show Grad-CAM heatmap highlighting regions of interest

## Future Development

- **Lung Segmentation**: Implement U-Net architecture for automatic Region of Interest (ROI) extraction
- **Improved Accuracy**: Focus model attention exclusively on lung parenchyma
- **Multi-class Classification**: Extend to detect specific pneumonia types (bacterial vs. viral)

## Academic Context

This project was developed as a course project for the Artificial Intelligence course at Sofia University "St. Kliment Ohridski", Faculty of Mathematics and Informatics (2025/26 academic year).

**Students**: Bogomil Boykov Stoyanov, Mihail Boykov Dobroslavski  
**Instructor**: Prof. Ivan Koychev

## Disclaimer

⚠️ **Medical Disclaimer**: This AI system is for educational and screening purposes only. It should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

## References

1. Kermany, D. S., et al. "Identifying medical diagnoses and treatable diseases by image-based deep learning." *Cell* 172.5 (2018): 1122-1131.
2. Simonyan, K., & Zisserman, A. "Very deep convolutional networks for large-scale image recognition." *ICLR* (2015).
3. Selvaraju, R. R., et al. "Grad-CAM: Visual explanations from deep networks via gradient-based localization." *ICCV* (2017).
4. Pan, S. J., & Yang, Q. "A survey on transfer learning." *IEEE TKDE* 22.10 (2010): 1345-1359.

## License

This project is open-source and available for educational purposes.
