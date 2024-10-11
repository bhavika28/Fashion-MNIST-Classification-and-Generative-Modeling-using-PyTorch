# **Fashion MNIST GAN - Generative Adversarial Network using PyTorch**

### **Project Description**
This project implements a **Generative Adversarial Network (GAN)** using the **Fashion MNIST dataset** in **PyTorch**. The goal is to generate realistic images of clothing items from random noise by training two neural networks: a **generator** and a **discriminator**. The generator tries to create fake images that look like real images from the dataset, while the discriminator attempts to differentiate between real and fake images.

### **Dataset**
The **Fashion MNIST** dataset consists of 28x28 grayscale images from 10 categories of clothing (e.g., T-shirts, shoes, coats). This dataset is widely used as a drop-in replacement for MNIST, providing a more challenging image classification task with a more diverse range of real-world items.

### **Architecture**

- **Discriminator**:
  - A fully connected neural network designed to classify images as real or fake.
  - Takes an image (flattened to 784 pixels) as input and outputs a probability (using Sigmoid) indicating whether the image is real or fake.
  - Contains three layers with LeakyReLU activations and a Sigmoid output.

- **Generator**:
  - A fully connected neural network designed to generate fake images from random noise (latent vector).
  - Takes a random noise vector of size 100 as input and produces a 784-dimensional vector (28x28 image).
  - Contains four layers with LeakyReLU activations, Batch Normalization, and Tanh output to generate images in the range [-1, 1].

### **Project Steps**

1. **Import Libraries**: Essential packages such as PyTorch, torchvision (for loading Fashion MNIST), matplotlib (for visualizations), and PyTorch's neural network modules (`torch.nn`).
2. **Load the Dataset**: The Fashion MNIST dataset is loaded and preprocessed using transformations like normalization to ensure pixel values are in the [-1, 1] range.
3. **Define Discriminator**: The discriminator network is built to classify whether images are real or fake.
4. **Define Generator**: The generator network is built to generate images from random noise (latent space).
5. **Generate Samples**: After defining both networks, samples of fake clothing images are generated and saved as PNG files.


### **Results**
You can find the generated images in the **generated_samples/** directory after running the notebook. The generator produces fake images of clothing that resemble items in the Fashion MNIST dataset.

### **Future Work**
- Implement the **training loop** for the GAN, allowing the generator and discriminator to learn and improve over time.
- Explore different architectures for the generator and discriminator, such as adding convolutional layers (CNNs) for improved performance.
- Experiment with **hyperparameter tuning** to improve the quality of the generated images.

### **Technologies Used**
- **Python**
- **PyTorch**
- **Torchvision**
- **Matplotlib**
