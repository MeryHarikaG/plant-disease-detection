# PhytoNet: Early Plant Disease Detection

PhytoNet is an advanced deep learning model developed to improve the detection of plant diseases at an early stage. By using Convolutional Neural Networks (CNNs), PhytoNet examines images of plant leaves to identify various diseases, greatly enhancing the traditional method of manual inspection.

## Features

- **Deep Learning Model**: Utilizes a CNN architecture to analyze and classify leaf images.
- **Data Handling**: Manages a dataset of over 70,000 images across 38 distinct classes, ensuring robust training and validation.
- **Performance Monitoring**: Employs techniques like ModelCheckpoint and EarlyStopping to optimize training outcomes.
- **High Accuracy**: Achieves impressive performance metrics, with a validation accuracy of 86.02% and a testing accuracy of 90.38%.

## Model Architecture

- **Input Layer**: Accepts 128x128 pixel images.
- **Convolutional Layers**: Four layers with increasing filter complexity (32, 64, 128, 256).
- **Pooling Layers**: Max pooling for spatial data reduction.
- **Dense Layers**: Includes a flattened layer and two dense layers ending with a softmax activation for classification.

## Getting Started

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- Keras
- NumPy, Matplotlib

### Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/MeryHarikaG/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
```

### Usage
Run the Jupyter notebook to train the model and evaluate its performance:

```bash
jupyter notebook ProjF5_Final_Update_Team_96.ipynb
```

### Future Work
- Exploring advanced data augmentation techniques.
- Investigating the use of transfer learning from pre-trained models.
- Developing applications for real-time plant disease detection using smartphones.
