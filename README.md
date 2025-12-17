# Mushroom Toxicity Classifier 🍄

A deep learning computer vision project for classifying mushroom sporocarps as edible or poisonous using PyTorch Lightning and ResNet50.

## 🚀 Live Demo

Try the model yourself: **[Hugging Face Space Demo](https://huggingface.co/spaces/matsuokengo/sporocarp-toxicity-classifier)**

## Overview

This project implements a binary image classifier to identify whether mushroom sporocarps (fruiting bodies) are edible or poisonous. The model leverages transfer learning with a pre-trained ResNet50 architecture, fine-tuned on a custom dataset of mushroom images.

## Features

- **Transfer Learning**: Uses ResNet50 pre-trained on ImageNet
- **Data Augmentation**: Includes random crops, flips, rotations, and color jittering
- **PyTorch Lightning**: Clean, modular training pipeline
- **Mixed Precision Training**: Automatic GPU acceleration with 16-bit precision
- **TensorBoard Logging**: Real-time training metrics visualization
- **Model Checkpointing**: Saves best models based on validation accuracy
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Comprehensive Evaluation**: Classification reports and confusion matrices

## Dataset

**Source**: [Kaggle - Edible and Poisonous Fungi Dataset](https://www.kaggle.com/datasets/marcosvolpato/edible-and-poisonous-fungi/data)

The dataset contains images of mushroom sporocarps organized into four categories:
- `edible sporocarp`
- `edible mushroom sporocarp`
- `poisonous sporocarp`
- `poisonous mushroom sporocarp`

These are mapped to binary labels: **edible (0)** and **poisonous (1)**.

## Project Structure

```
computer-vision-aol/
├── train.ipynb                 # Main training notebook
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── best_fungi_model.pt         # Best trained model
├── model.pt                    # Latest model checkpoint
├── dataset/                    # Training images
│   ├── edible sporocarp/
│   ├── edible mushroom sporocarp/
│   ├── poisonous sporocarp/
│   └── poisonous mushroom sporocarp/
├── checkpoints/                # Model checkpoints
│   ├── fungi-epoch=*.ckpt
│   └── last.ckpt
└── lightning_logs/             # TensorBoard logs
    └── fungi_classifier/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kengomatsuo/computer-vision-aol.git
cd computer-vision-aol
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Open and run `train.ipynb` in Jupyter Notebook or VS Code:

1. **Data Loading**: Loads images from the dataset folder
2. **Data Splitting**: 80/20 train/validation split with stratification
3. **Data Augmentation**: Applies transformations to training data
4. **Model Training**: Trains ResNet50 with PyTorch Lightning
5. **Model Evaluation**: Generates classification report and confusion matrix

### Hyperparameters

Key training parameters (configurable in notebook):

```python
IMG_SIZE = 224              # Input image size
BATCH_SIZE = 32             # Batch size
NUM_EPOCHS = 20             # Training epochs
LEARNING_RATE = 1e-4        # Initial learning rate
WEIGHT_DECAY = 0.01         # L2 regularization
TEST_SPLIT = 0.2            # Validation split ratio
MODEL_NAME = 'resnet50'     # Backbone architecture
```

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir lightning_logs/
```

## Model Architecture

- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Input Size**: 224×224×3
- **Output**: 2 classes (edible, poisonous)
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: AdamW with weight decay
- **Scheduler**: ReduceLROnPlateau

## Data Augmentation

**Training Transforms**:
- Resize to 256×256
- Random crop to 224×224
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- ImageNet normalization

**Validation Transforms**:
- Resize to 224×224
- ImageNet normalization

## Results

The model achieves:
- **Best Validation Accuracy**: ~87.4% (based on checkpoints)
- Saved checkpoints include epoch-specific validation accuracies
- Detailed per-class metrics available in classification report

## Model Checkpointing

The training pipeline saves:
- **Top 3 models** based on validation accuracy
- **Last checkpoint** for resuming training
- **Best model** exported as `best_fungi_model.pt`

Checkpoint naming: `fungi-epoch={epoch:02d}-validation_accuracy={val_acc:.3f}.ckpt`

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- torchvision
- timm (PyTorch Image Models)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- opencv-python
- albumentations

See `requirements.txt` for complete dependencies.

## Callbacks & Features

- **ModelCheckpoint**: Saves best models based on validation accuracy
- **EarlyStopping**: Stops training if validation loss doesn't improve (patience=7)
- **LearningRateMonitor**: Tracks learning rate changes
- **Mixed Precision**: Automatic FP16 training on GPU
- **Per-Class Accuracy**: Monitors accuracy for both edible and poisonous classes

## Evaluation Metrics

The final evaluation includes:
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Visual heatmap of predictions
- **Per-Class Accuracy**: Separate metrics for edible and poisonous mushrooms

## Safety Warning ⚠️

**This model is for educational purposes only.** Never rely on automated systems to determine mushroom edibility in real-world scenarios. Consuming poisonous mushrooms can be fatal. Always consult expert mycologists and use proper identification methods.

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: [Marcos Volpato on Kaggle](https://www.kaggle.com/datasets/marcosvolpato/edible-and-poisonous-fungi)
- ResNet50: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- PyTorch Lightning: [Lightning AI](https://lightning.ai/)

## Author

Created as part of a computer vision project.