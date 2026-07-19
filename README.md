
#  Fashion Classifier

A deep learning model trained on **40,000+ images** that classifies clothing images into 10 categories using a multi-layer neural network built with TensorFlow/Keras — with a live interactive web app powered by Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Output

<!-- To upload screenshots: go to your GitHub repo → open any Issue → drag and drop your screenshot into the comment box → copy the generated image URL → paste it in place of the demo/ paths below -->

| Input Image | Predicted Class | Confidence |
|---|---|---|
<img width="1649" height="775" alt="image" src="https://github.com/user-attachments/assets/5853bcdd-5369-42d5-bac8-40d85be2fb46" />


---

## 🎯 What It Does

Upload any clothing image and the model predicts which of the 10 categories it belongs to — along with a confidence score.

**Supported Categories:**

| | | |
|---|---|---|
| 👟 Ankle Boot | 👜 Bag | 🧥 Coat |
| 👗 Dress | 🧥 Pullover | 👡 Sandal |
| 👔 Shirt | 👟 Sneaker | 👕 T-shirt |
| 👖 Trouser | | |

---

## 🧠 Model Architecture

Built using a fully connected feedforward neural network (no CNNs — pure dense layers):

```
Input (28×28 grayscale image)
    ↓
Rescaling (÷ 255 — normalise pixel values to [0, 1])
    ↓
Flatten → 784 neurons
    ↓
Dense(128, ReLU)
    ↓
Dense(64, ReLU)
    ↓
Dense(32, ReLU)
    ↓
Dense(10, Softmax) → 10 class probabilities
```

**Training details:**
- Optimiser: Adam
- Loss function: Sparse Categorical Crossentropy
- Epochs: 70
- Batch size: 32
- Training images: 40,000+
- Input: 28×28 grayscale images

---

## 📁 Project Structure

```
fashion-classifier/
│
├── fashion_classifier.py   # Model training script
├── app.py                  # Streamlit web app
├── fashion_classifier.keras  # Saved trained model
├── dataset/
│   ├── train/              # Training images (organised by class folder)
│   └── test/               # Test images (organised by class folder)
└── README.md
```

---

## 🚀 Getting Started

###
---

## 🗂️ Dataset Format

The dataset should be organised as follows (one folder per class):

```
dataset/
├── train/
│   ├── Ankle Boot/
│   ├── Bag/
│   ├── Coat/
│   ├── Dress/
│   ├── Pullover/
│   ├── Sandal/
│   ├── Shirt/
│   ├── Sneaker/
│   ├── T-shirt/
│   └── Trouser/
└── test/
    └── (same structure as train/)
```

Images should be **grayscale**, **28×28 pixels**. The model uses `tf.keras.utils.image_dataset_from_directory` to load them automatically from this folder structure.

---

## 💡 Key Concepts Used

- **Rescaling layer** — normalises pixel values from [0, 255] → [0, 1] for faster convergence
- **Flatten layer** — converts 2D image (28×28) into a 1D vector (784) as input to dense layers
- **ReLU activation** — introduces non-linearity in hidden layers
- **Softmax activation** — converts final layer outputs into class probabilities (sums to 1.0)
- **Sparse Categorical Crossentropy** — appropriate loss function for multi-class classification with integer labels
- **Adam optimiser** — adaptive learning rate optimiser, efficient for this type of classification task

---

## 🔍 How the App Works

1. User uploads a `.jpg`, `.jpeg`, or `.png` image
2. App converts it to grayscale and resizes to 28×28 pixels
3. Image is normalised and reshaped to match model input `(1, 28, 28)`
4. Model outputs 10 probability values via Softmax
5. App displays the predicted class and confidence score

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model building and training
- **Streamlit** — interactive web interface
- **NumPy** — array manipulation
- **Pillow (PIL)** — image preprocessing

---

## 👤 Author

**Varad Tushar Gaikwad**
- GitHub: [@Varad-gaikwad](https://github.com/Varad-gaikwad)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
