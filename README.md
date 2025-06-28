# 🧠 DCGAN for Handwritten Digit Generation (MNIST)

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate **fake handwritten digits** similar to those in the MNIST dataset. The model learns the distribution of real handwritten digits and generates realistic samples from random noise inputs.



## 📑 Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



## 📝 Introduction

Generative Adversarial Networks (GANs) are an exciting class of deep learning models capable of generating new data samples. In this project, we build a **DCGAN** to generate images resembling handwritten digits from the **MNIST dataset** using PyTorch.



## 📊 Dataset

* **Dataset:** [MNIST]()
* **Description:** Contains 70,000 grayscale images of handwritten digits (0-9), each of size **28x28 pixels**.



## 🏗️ Model Architecture

### Generator

* Takes random noise (latent vector) as input
* Uses transposed convolution layers to upsample to **28x28** images
* Outputs images with pixel values in the range \[-1, 1] (tanh activation)

### Discriminator

* Takes a real or fake image as input
* Uses convolutional layers with LeakyReLU activations to downsample
* Outputs a single scalar indicating real or fake (Sigmoid activation)

The generator and discriminator are trained adversarially:

* **Generator:** Learns to create realistic digits to fool the discriminator
* **Discriminator:** Learns to distinguish between real and fake digits



## ✨ Technologies Used

* **Python 3**
* **PyTorch**
* `torchvision` (for dataset loading)
* `numpy`
* `matplotlib` (for visualization)
* **Jupyter Notebook**



## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/DCGAN-for-Handwritten-Digit-Generation-MNIST.git
cd DCGAN-for-Handwritten-Digit-Generation-MNIST
```

2. **Create and activate a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```



## ▶️ Usage

1. Open `DCGAN_MNIST.ipynb` in Jupyter Notebook.
2. Run cells sequentially to:

   * Import libraries and load dataset
   * Define the **Generator** and **Discriminator** models
   * Train the GAN for a set number of epochs
   * Visualize generated samples after training



## 📁 Project Structure

```
DCGAN-for-Handwritten-Digit-Generation-MNIST/
 ┣ data/
 ┃ ┗ (MNIST dataset downloaded automatically by torchvision)
 ┣ images/
 ┃ ┗ (Generated sample images)
 ┣ DCGAN_MNIST.ipynb
 ┣ requirements.txt
 ┗ README.md
```



## 📈 Results

Below are some sample **generated digits** after training:

![Generated Digits](images/generated_digits.png)

* The DCGAN learns to generate digits that resemble MNIST samples after sufficient epochs.
* Generation quality improves with training stability and hyperparameter tuning.



## 🤝 Contributing

Contributions are welcome to:

* Improve training stability (e.g. Wasserstein GAN)
* Generate coloured digits or apply to other datasets
* Experiment with different latent space sizes and architectures

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request



## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



## 📬 Contact

**Ugama Benedicta Kelechi**
[LinkedIn](www.linkedin.com/in/ugama-benedicta-kelechi-codergirl-103041300) | [Email](mailto:ugamakelechi501@gmail.com) | [Portfolio Website](#)



### ⭐️ If you find this project useful, please give it a star!


