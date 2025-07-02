# COMSYS Hackathon 2025 - Task B: Robust Face Recognition

This project implements a robust face recognition system using deep learning (ResNet50V2) that can identify people even under various image distortions (blur, fog, low light, noise, etc.).

## Features
- **Transfer Learning** with ResNet50V2
- **Advanced Data Augmentation** for robustness
- **Mixed Precision Training** for speed
- **Comprehensive Evaluation** and visualization
- **Easy Inference** on single images or validation samples

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/agnihotri-anxh/COMSYS-Hackathon-50-2025-Task_B.git
   cd COMSYS-Hackathon-50-2025-Task_B
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   Or manually:
   ```sh
   pip install numpy tensorflow matplotlib pandas scikit-learn opencv-python pillow seaborn jupytext
   ```

3. **Prepare the dataset:**
   - Place your dataset in `dataset/Task_B/train/` as per the folder structure.
   - **Note:** The `dataset/`, `models/`, and all `.npy` files are excluded from git tracking (see `.gitignore`).

## Training

To train the model:
```sh
python face_trainer.py
```
- This will save the best model, label encoder, and training logs in the `models/` and `results/` folders.

## Evaluation

To evaluate the trained model:
```sh
python evaluate_model.py
```
- This will print accuracy, F1-score, and generate plots.

## Inference

To predict on new images or validation samples:
- **Single image:**
  ```sh
  python predict_faces.py --image path/to/image.jpg
  ```
- **Random validation images:**
  ```sh
  python predict_faces.py --val-random 5 --visualize
  ```
- **Interactive mode:**
  ```sh
python predict_faces.py --interactive
```

## License
MIT License

---
