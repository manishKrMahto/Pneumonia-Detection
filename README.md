# Pneumonia Detection 
#### ***from Chest X-rays using CNN and Transfer Learning***

This repository presents a complete deep learning pipeline to detect **Pneumonia** from chest X-ray images. The project uses **Convolutional Neural Networks (CNN)**, **VGG19 Transfer Learning**, and a **Streamlit-based web app** to provide an end-to-end solution—from data preprocessing and model training to real-time predictions.

---

## 📁 Project Structure

- **`Pneumonia_detection_using_CNN.ipynb`** – Custom CNN model training and evaluation.
- **`Pneumonia_detection_using_VGG19.ipynb`** – Transfer learning using VGG19 with frozen base layers.
- **`Pneumonia_detection_using_VGG19_Fine_Tuning.ipynb`** – Fine-tuned VGG19 with unfreezing of last 4 convolution blocks.
- **`app.py`** – Streamlit-based web app to detect Pneumonia from uploaded X-ray images.

---

## 📊 Dataset

- **Source**: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Directory Structure**:
```

chest_xray/
├── train/
│   ├── NORMAL (1341 images)
│   └── PNEUMONIA (3875 images)
├── test/
│   ├── NORMAL (234 images)
│   └── PNEUMONIA (390 images)
└── val/
├── NORMAL (8 images)
└── PNEUMONIA (8 images)

````

## First 16 Pneumonia x-rays
<img src = "images/first 16 Pneumonia images.png" alt = "first 16 pneumonia images from train folder"></img>

---

## First 16 Normal x-rays
<img src = "images/first 16 Normal images.png" alt = "first 16 Normal images from train folder"></img>



- The dataset is **imbalanced**, with significantly more Pneumonia cases than Normal.
<img src = "images/train imbalance classes.png" alt ="train imbalance class"></img>
---

## 🧪 Preprocessing and Augmentation

Used `ImageDataGenerator` with the following transformations:

````python

ImageDataGenerator(
  rescale=1./255,
  horizontal_flip=True,
  vertical_flip=True,
  rotation_range=40,
  shear_range=0.2,
  width_shift_range=0.4,
  height_shift_range=0.4,
  fill_mode="nearest"
)

````

Images resized to **128x128** and loaded using `flow_from_directory`.

---

## 🧠 Model Architectures

### 1. Custom CNN Model

* 4 Convolutional Blocks with BatchNorm + MaxPooling
* Fully connected layers: 512 → 128 → Output(2)
* Regularization: Dropout(0.3)
* **EarlyStopping** and **ReduceLROnPlateau** used for training control
* **5.17M+** parameters

**Accuracy**:

* Train: 79.17%
* Val: 56.25%
* Test: 73.85%

**CNN History**
<img src = "images/CNN history.png" alt ="CNN history"></img>

---

### 2. VGG19 (Transfer Learning)

* Base model: VGG19 with frozen layers
* Custom Dense Head: 4096 → 2048 → 1024 → Output(2)
* Dropout(0.2) between layers
* **64M+** parameters

**Accuracy**:

* Train: 92.71%
* Val: 75.00%
* Test: 82.44%

**VGG19 History**
<img src = "images/VGG19 dense history.png" alt ="VGG19 dense history"></img>

---

### 3. VGG19 (Fine-Tuned)

* Unfrozen last 4 convolutional layers: `block5_conv1` to `block5_conv4`
* Retained same dense structure as above

**Accuracy**:

* Train: 96.41%
* Val: 93.75%
* Test: 92.25%

**CNN History**
<img src = "images/VGG19 Fine-Tune history.png" alt ="VGG19 Fine-Tune history"></img>



---

## 🌐 Streamlit Web App

A clean and simple web UI to test X-ray predictions using the **Fine-Tuned VGG19 model**.

### Features:

* Upload `.jpg`, `.jpeg`, or `.png` X-ray images
* Predicts **Pneumonia** or **Normal**
* Displays prediction confidence

### Example Output:

````
Prediction: PNEUMONIA
Confidence: 95.23%
````

<img src ="images/frontend.jpg" alt = "frontend streamlit picture"></img>

---

## ▶️ How to Run

### 1. Clone the Repository

````bash
git clone https://github.com/manishKrMahto/Pneumonia-Detection.git
cd Pneumonia-Detection
````

### 2. Install Dependencies

````bash
pip install -r requirements.txt
````

**`requirements.txt` includes:**
> `tensorflow`, `streamlit`, `opencv-python`, `numpy`, `matplotlib`, `Pillow`, `scikit-learn`

### 3. Run the Streamlit App

````bash
streamlit run app.py
````

---

## 🔧 Libraries Used

* `TensorFlow`, `Keras`
* `OpenCV`, `NumPy`, `Matplotlib`
* `Streamlit` (for web app)
* `scikit-learn` (for classification metrics)

---

## 📈 Evaluation Metrics

* **Accuracy**
* **Loss**


### Each model was trained using:

* `Adam` Optimizer
* `EarlyStopping` (monitoring `val_loss`)
* `ReduceLROnPlateau` (`min_lr=1e-4`)

---

## 📬 Contact

Created by **Manish Kumar Mahto** – Feel free to connect on [LinkedIn](https://www.linkedin.com/in/manish-kumar-mahto/) for questions or collaborations!

