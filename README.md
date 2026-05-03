# Fashion Product Image Classification: Addressing Long-Tail Data Imbalance

## 📌 Project Goal
The primary objective of this project is to develop a robust multi-class image classification model capable of accurately categorizing low-resolution fashion product images. Specifically, this project aims to solve the extreme **long-tail data imbalance problem** inherent in fashion datasets. By implementing an `EfficientNet-V2-S` architecture with transfer learning and integrating a **Class-Balanced Loss (Class-Weighted Loss)** strategy, we successfully improved the model's recall for minority classes without severely sacrificing overall global accuracy.

## 📊 Dataset Information
* **Dataset Name:** `ashraq/fashion-product-images-small` (Loaded via HuggingFace `datasets` library)[cite: 1, 5].
* **Task:** Multi-class image classification predicting the `articleType`[cite: 1, 5].
* **Classes:** 141 unique sub-categories[cite: 1, 5].
* **Data Split:** The dataset is split into Train, Validation, and Test sets using a strict **6:2:2 Stratified Split** to ensure minority classes are proportionally represented across all phases.

## 🛠 Environment and Dependencies
This project was developed and evaluated using PyTorch. 

**Core Dependencies:**
* `torch`, `torchvision` (Model architecture and training)[cite: 1, 5]
* `datasets` (HuggingFace dataset loading)[cite: 1, 5]
* `gradio` (Inference web demo)
* `matplotlib`, `seaborn` (Visualization of training curves and confusion matrices)[cite: 1, 5]
* `scikit-learn` (Metrics computation)[cite: 1, 5]
* `numpy`, `Pillow`[cite: 1, 3]

**Installation:**
```bash
pip install torch torchvision datasets gradio matplotlib seaborn scikit-learn numpy pillow
```

## 4. Training Instructions
To train the EfficientNet-V2-S model with Class-Weighted Loss and Custom Data Augmentation, run the `train.py` script. The script automatically downloads the Hugging Face dataset and saves the weights as `best_model.pth`.
```bash
python train.py --batch_size 32 --epochs 20 --lr 0.001
```

## 5. Evaluation Instructions
To evaluate the trained model on the test dataset and generate visual performance metrics (such as the 141x141 Confusion Matrix, Minority/Majority class graphs), run:
```bash
python evaluate.py --model_path ./best_model.pth
```

## 6. How to Run the Inference Demo
We provide an inference script to test the model on individual images. You can pass a sample image to see the model's top predicted fashion category and its confidence score.
```bash
python inference_demo.py --model_path ./best_model.pth --image_path ./sample_image.jpg
```
