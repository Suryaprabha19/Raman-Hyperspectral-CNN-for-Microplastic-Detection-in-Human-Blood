# Raman Hyperspectral CNN for Microplastic Detection in Human Blood

A deep learning-based application for detecting microplastics in blood samples using Raman spectroscopy. This project uses a U-Net architecture to segment and classify plastic particles from hyperspectral Raman cube data.

## 🔬 Overview

This project processes Raman spectroscopy data to identify and classify microplastic contamination in blood samples. It combines hyperspectral imaging (100-band Raman cubes) with microscope imagery to provide accurate detection and classification of five common plastic types:

- **PET** (Polyethylene Terephthalate) - Water bottles
- **PS** (Polystyrene) - Food containers
- **PE** (Polyethylene) - Plastic bags
- **PP** (Polypropylene) - Bottle caps
- **PMMA** (Polymethyl Methacrylate) - Cosmetics

## 📁 Project Structure

```
Raman/
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── data_factory.py             # Dataset generation utility
├── generate_report.py          # Report generation module
├── raman_model.pth             # Trained model weights
└── raman_blood_dataset/
    ├── microscope_images/      # RGB microscope images
    ├── raman_cubes/            # Hyperspectral Raman data (128×128×100)
    ├── ground_truth_masks/     # Segmentation masks
    └── metadata/               # Sample metadata (JSON)
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit numpy matplotlib torch torchvision
```

### Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🧪 Usage

### 1. Training the Model

Generate synthetic dataset and train the model:

```bash
# Generate synthetic Raman dataset
python data_factory.py

# Train the U-Net model
python train_model.py
```

The trained model will be saved as `raman_model.pth`.

### 2. Web Application Features

The Streamlit app provides:

- **Random Sample Loading**: Automatically loads a random blood sample from the dataset
- **Microscope Image Display**: Shows the RGB microscope view of the sample
- **Raman Spectrum Visualization**: Displays the spectral signature at clicked points
- **Automated Detection**: AI-powered plastic detection and classification
- **Physical Properties**: Shows density, color, and likely source of detected plastics
- **Interactive Analysis**: Click on the image to examine Raman spectra at specific locations

## 🔬 Dataset Format

### Microscope Images
- Format: PNG
- Size: 128×128 pixels
- Type: RGB color images

### Raman Cubes
- Format: NumPy (.npy)
- Shape: 128×128×100
  - 128×128: Spatial dimensions
  - 100: Spectral bands (Raman wavenumbers)

### Ground Truth Masks
- Format: PNG
- Size: 128×128 pixels
- Values: Binary (0 = blood, 1 = plastic)

### Metadata
- Format: JSON
- Contains: Patient ID, sample type, contamination info

## 🧠 Model Architecture

**RamanUNet**: A simplified U-Net architecture designed for hyperspectral segmentation

- **Input**: 100-channel Raman cube (100×128×128)
- **Architecture**:
  - Encoder 1: Conv2d(100→64) + ReLU
  - Encoder 2: Conv2d(64→128) + ReLU
  - Output: Conv2d(128→2) for binary classification
- **Output**: 2-class segmentation (blood vs plastic)

## 📊 Physical Properties Database

Each detected plastic type includes:
- **Density**: Material density in g/cm³
- **Color**: Visual appearance characteristics
- **Common Source**: Typical origin of contamination

## 🛠️ Technical Details

- **Framework**: PyTorch for deep learning
- **UI**: Streamlit for web interface
- **Visualization**: Matplotlib for plotting
- **Data Format**: NumPy arrays for efficient processing

## 📈 Training Parameters

- **Batch Size**: 4
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: Configurable in `train_model.py`

## 🎯 Key Features

1. **Hyperspectral Analysis**: Utilizes 100-band Raman spectroscopy for accurate material identification
2. **Real-time Detection**: Fast inference on new samples
3. **Interactive Visualization**: Click-to-analyze interface for detailed spectral examination
4. **Physical Characterization**: Automatic lookup of material properties
5. **Synthetic Data Generation**: Built-in dataset generator for training and testing

