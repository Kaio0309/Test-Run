# MNIST Classification with CI/CD

![ML Pipeline](https://github.com/YOUR-USERNAME/Test-Run/actions/workflows/ml-pipeline.yml/badge.svg)

## Overview
This project implements a CNN-based MNIST classifier with complete CI/CD pipeline using GitHub Actions. The model is a lightweight CNN with less than 25,000 parameters that achieves >95% accuracy on MNIST.

## Model Architecture
- 2 Convolutional layers with max pooling
- 2 Fully connected layers
- Dropout for regularization
- Total parameters: ~13,242

## CI/CD Pipeline Features
- Automated model training
- Parameter count verification (<25,000)
- Input shape validation (28x28)
- Output dimension check (10 classes)
- Accuracy testing (>95%)
- Automated model artifact storage

## Local Development
1. Create virtual environment:
