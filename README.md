# Pneumonia Detection from Chest X-Rays

Automated pneumonia diagnosis using deep learning to assist healthcare professionals in rapid and accurate detection.

## Problem

Pneumonia is a life threatening respiratory infection that affects millions worldwide, particularly children and elderly populations. The key challenges in pneumonia diagnosis include:

- **Time-Critical Diagnosis**: Manual interpretation of chest X-rays requires specialized radiological expertise
- **Resource Constraints**: Limited availability of trained radiologists, especially in underserved areas
- **Human Error**: Fatigue and workload can impact diagnostic accuracy
- **Scalability**: Growing patient volumes require faster, more efficient diagnostic tools

This project addresses these challenges by developing an AI powered system that can automatically classify chest X-ray images as **NORMAL** or **PNEUMONIA**, providing rapid preliminary assessments to support clinical decision making.

## Approach

My solution implements a comprehensive deep learning pipeline designed specifically for medical image classification:

### Data Processing Strategy
- **Image Standardization**: Resize all X-rays to 224×224 pixels for consistent input
- **Normalization**: Pixel value scaling for optimal neural network performance  
- **Data Augmentation**: Rotation, scaling, and flipping to improve model generalization
- **Class Balance Handling**: Computed class weights to address dataset imbalance

### Model Architecture
- **Convolutional Layers**: Multiple Conv2D layers for feature extraction from X-ray images
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Pooling Operations**: MaxPooling reduces spatial dimensions while preserving important features
- **Dense Classification**: Fully connected layers for final binary classification
- **Regularization**: Dropout and other techniques to prevent overfitting

### Evaluation Framework
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score
- **Visual Analysis**: Confusion matrices and classification reports
- **Clinical Relevance**: Focus on minimizing false negatives (missed pneumonia cases)


## How to Run

### Prerequisites
- Python 3.8 or higher


### Installation
```bash
# Clone the repository
git clone https://github.com/Ramiassaf/CHESTXRAYPROJECT
cd CHESTXRAYPROJECT

# Install required dependencies 
pip install -r requirements.txt
```
# Option 1: Open in VS Code (if you have Jupyter extension)
code main.ipynb

# Option 2: Run with Jupyter Notebook
jupyter notebook main.ipynb


### Tests
Run from the project root:
```bash
pytest -v
```


### Test Coverage
- **Model Loading**: Verify `cnn_pneumonia_model.keras` loads correctly
- **Data Pipeline**: Test preprocessing with your `data/train`, `data/val`, `data/test` folders
- **Sample Predictions**: Validate predictions on `person1_virus_12.jpeg` and other sample images



## Results

My CNN model demonstrates strong diagnostic performance on the chest X-ray test dataset:

### Performance Metrics

| Metric | Score | Clinical Significance |
|--------|-------|---------------------|
| **Overall Accuracy** | 88% | Strong general performance |
| **Pneumonia Precision** | 95% | Low false positive rate |
| **Pneumonia Recall** | 86% | Catches most pneumonia cases |
| **F1-Score** | 0.90 | Balanced precision-recall |

### Detailed Analysis

**Confusion Matrix**:
```
Predicted:     NORMAL  PNEUMONIA
Actual: NORMAL   [216     18]     # True Negatives: 216, False Positives: 18
    PNEUMONIA    [ 55    335]     # False Negatives: 55, True Positives: 335
```

### Clinical Interpretation
- **High Specificity (92%)**: Excellent at correctly identifying healthy patients
- **Strong Sensitivity (86%)**: Successfully detects most pneumonia cases  
- **Low False Positive Rate (8%)**: Minimizes unnecessary treatments
- **Moderate False Negative Rate (14%)**: Area for improvement to catch all cases

The model prioritizes patient safety with conservative predictions while significantly reducing radiologist workload for routine screenings.

### Visual Results
- Loss reduction over training epochs
- Confusion matrix 

## Roadmap
 - Build & train baseline CNN model

 - Evaluate with confusion matrix

 - Save trained model (.keras & .h5)


## Project Structure
```
CHESTXRAYPROJECT/
├── data/                           # Dataset (train/val/test)
├── cnn_pneumonia_model.h5          # Trained model (HDF5 format)
├── cnn_pneumonia_model.keras       # Trained model (Keras format)
├── main.ipynb                      # Complete pipeline notebook
├── requirements.txt                # Dependencies
├── Makefile                        # Common automation tasks
├── README.md                       # Project documentation
├── training_log.csv                # Training logs
│
├── tests/                          # Automated tests
│   ├── test_model.py               # Pytest file (model existence & loading)
│   └── __pycache__/                # Auto-generated Python cache (ignored)
│
└── .pytest_cache/                  # Auto-generated pytest cache (ignored))
```


## Dataset & Acknowledgments

**Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) - Kaggle

⚠️ Note: Model files (`.keras` and `.h5`) are too large for GitHub (>100 MB) and are excluded from the repository.  
They will be generated when you train the model using the notebook.

