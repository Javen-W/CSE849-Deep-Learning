# CSE849: Deep Learning Coursework
This repository contains my coursework for CSE849, a graduate-level Deep Learning course completed as part of my Master’s in Computer Science and Engineering. It includes four projects and one theoretical assignment, demonstrating my proficiency in designing, implementing, and analyzing deep learning models. The projects emphasize PyTorch, Convolutional Neural Networks (CNNs), Transformers, Natural Language Processing (NLP), and NumPy, showcasing my readiness for machine learning engineering roles.

## Projects

### Project 0: Familiarizing with PyTorch
**Description**: Developed a PyTorch pipeline from scratch for a regression task, generating a synthetic dataset and training a linear model, then applied a multi-layer perceptron (MLP) to a provided dataset (`hw0/CSE_849___Project_0.pdf`).

**Approach**: Created a custom Dataset class to generate synthetic data using the formula `y = αx + βe`, with random seed control for reproducibility. Built a DataLoader for efficient batching. Implemented a linear model with nn.Linear and Xavier initialization, trained with MSE loss and AdamW optimizer. For the provided data.pt dataset, trained an MLP with two hidden layers (`1D -> 10D -> 10D -> 1D`) using AdamW, visualizing predictions across multiple seeds.

**Tools**: PyTorch, NumPy, Matplotlib.

**Results**: Achieved low training and validation MSE for the synthetic dataset, with visualized loss curves (`hw0/results/q2_plot.png`). For the provided dataset, tuned hyperparameters (batch size, learning rate) to optimize validation performance, with predictions plotted for seeds 1-5 (`hw0/results/q3_plot.png`).

![q2_plot](https://github.com/user-attachments/assets/07fff7b7-fde5-4ba9-9889-fc44de6e8483)

![q3_plot](https://github.com/user-attachments/assets/8042ba73-e87c-44e6-8695-ca5fafa8b22f)

**Key Skills**: PyTorch fundamentals, PyTorch pipeline development, custom dataset creation, NumPy data generation, MLP implementation.

### Project 1: Backpropagation
**Description**: Developed a neural network to explore backpropagation for regression on a synthetic dataset, focusing on gradient computation (hw1/CSE_849___Project_1.pdf).

**Approach**: Built a three-layer network in PyTorch with custom backpropagation logic to compute gradients manually and via autograd for comparison. Used NumPy to generate and preprocess data.

**Tools**: PyTorch, NumPy.

**Results**: Reduced mean squared error to 0.02 on the test set (hw1/results/loss_plot.png), validating accurate backpropagation implementation.

**Key Skills**: Backpropagation, PyTorch autograd, NumPy matrix operations.

### Homework 2: Derivatives for Batch Normalization
**Description**: Derived gradients for batch normalization to understand its role in stabilizing deep learning models (hw2/CSE_849___Project_2.pdf).

**Approach**: Provided mathematical proofs for the forward and backward passes of batch normalization, detailing variance and mean computations.

**Tools**: None (theoretical assignment).

**Results**: Successfully derived gradients, enhancing understanding of normalization techniques critical for CNNs and other architectures.

**Key Skills**: Mathematical foundations of deep learning, batch normalization theory.

### Project 2: Convolutional Neural Networks
**Description**: Designed a CNN to classify images from the CIFAR-10 dataset, exploring convolutional architectures (hw3/CSE_849___Project_2.pdf).

**Approach**: Constructed a CNN with PyTorch, using three convolutional layers, ReLU activations, max pooling, and batch normalization. Preprocessed images with NumPy for augmentation (e.g., random flips). Optimized with Adam.

**Tools**: PyTorch, NumPy, torchvision.

**Results**: Attained 82.3% test accuracy (hw3/results/accuracy_plot.png), with confusion matrix analysis to identify class-specific performance.

**Key Skills**: CNN architecture design, PyTorch implementation, NumPy preprocessing.

### Project 3: Sequence Modeling
**Description**: Built a Transformer-based model for sentiment analysis on the IMDB dataset, addressing NLP challenges (hw4/CSE_849___Project_3.pdf).

**Approach**: Implemented a Transformer encoder in PyTorch with multi-head self-attention and positional encodings. Preprocessed text with NumPy and tokenization to create word embeddings. Trained with cross-entropy loss.

**Tools**: PyTorch, NumPy, NLTK.

**Results**: Achieved 88.7% accuracy and 0.85 F1-score (hw4/results/metrics.txt), demonstrating robust sequence modeling for NLP tasks.

**Key Skills**: Transformers, NLP, sequence modeling, PyTorch.

## Skills Demonstrated
**Deep Learning**: Designed and trained advanced architectures, including Convolutional Neural Networks (CNNs) for image classification and Transformers for NLP tasks like sentiment analysis. Implemented backpropagation manually and explored batch normalization theoretically, ensuring a strong foundation in neural network mechanics.

**PyTorch Proficiency**: Leveraged PyTorch extensively to build, train, and evaluate models across all projects. Used PyTorch’s tensor operations and autograd for efficient gradient computation in Projects 0 and 1, constructed CNNs with dynamic layer configurations in Project 2, and implemented Transformer encoders with attention mechanisms in Project 3. Utilized PyTorch’s optimizers (SGD, Adam) and loss functions to optimize model performance, achieving high accuracy and stable training.

**Libraries and Tools**:
- NumPy: Applied for data preprocessing, including normalization and batching in Project 0, synthetic data generation in Project 1, and image augmentation (e.g., random flips) in Project 2. Used NumPy arrays for efficient text tokenization in Project 3.

- torchvision: Employed for dataset loading (e.g., CIFAR-10) and image transformations in Project 2, streamlining CNN development.

- NLTK: Utilized for text preprocessing and tokenization in Project 3, enabling effective NLP pipelines for sentiment analysis.

- Matplotlib: Created visualizations like loss curves and accuracy plots in Project 0, enhancing model interpretability.

**NLP Capabilities**: Developed Transformer-based models for sequence modeling, incorporating multi-head self-attention and positional encodings to handle text data (IMDB dataset). Processed text with tokenization and embeddings, achieving strong performance in sentiment analysis tasks.

**Technical Proficiency**: Combined theoretical insights (e.g., batch normalization derivations) with practical implementation, building robust ML pipelines. Demonstrated ability to preprocess diverse data types (images, text) and optimize models for real-world applications, aligning with machine learning engineering demands.



