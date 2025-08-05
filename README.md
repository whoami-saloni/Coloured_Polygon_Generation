# 🧠 Colored Polygon Generation using UNet

This project is part of the Ayna ML Internship Assignment. The objective is to train a deep learning model that can generate an image of a polygon filled with a given color based on:
- An **input polygon image** (uncolored)
- A **textual color name** (e.g., "red", "blue")

The output is an RGB image with the polygon filled with the specified color. The core model is a **UNet architecture**, enhanced to handle multi-modal inputs (image + text).


---

## 🧠 Model Architecture

- Base: **UNet**, built from scratch using PyTorch.
- Conditioning: The model is conditioned on a **color embedding**, generated from the textual color name.
- Integration:
  - Color name → index → learned embedding vector
  - This vector is concatenated and broadcasted across spatial dimensions and added to the encoder/decoder layers to condition the image generation.

---

## ⚙️ Training Details

- **Framework**: PyTorch
- **Loss**: MSELoss between predicted and target colored images
- **Optimizer**: Adam (lr=1e-4)
- **Epochs**: 30
- **Batch size**: 16
- **Augmentation**: Basic paired augmentations (random horizontal flip, random rotation)
- **Hardware**: Trained on T4 GPU (Google Colab)

### Hyperparameter Rationale

| Parameter      | Value     | Rationale                                                  |
|----------------|-----------|-------------------------------------------------------------|
| `lr`           | 1e-4      | Standard for stable convergence in image-to-image tasks     |
| `batch_size`   | 16        | Balanced for GPU memory constraints                        |
| `color_emb_dim`| 32        | Sufficient to capture basic color semantics                |
| `loss`         | MSE       | Suitable for pixel-level reconstruction                    |

---

## 📈 Training Dynamics

- **WandB Report Link **: [Click here to view](https://wandb.ai/salonisahal15-university-of-kalyani/colored-polygon-unet/reports/Report--VmlldzoxMzg3Mjg3MQ)
- **Metrics Logged**:
  - Training Loss
  - Validation Loss
- **Observations**:
  - Sharp loss decline during first 10 epochs.
  - Saturation observed near epoch 25.
  - Qualitative improvement in color accuracy and boundary sharpness over time.

---

## 🧪 Inference Notebook

- Provided in: [`Untitled31.ipynb`](./Untitled31.ipynb)
- Demonstrates:
  - Model loading
  - Sample inference with user-specified polygon and color
  - Visualization of input, color condition, and output side-by-side

---

## ❌ Common Failure Modes

| Issue                        | Mitigation Tried                               |
|-----------------------------|-------------------------------------------------|
| Slight color mismatch       | Increased embedding size, added batch norm     |
| Jagged polygon edges        | Applied smoother interpolation, denoising      |
| Overfitting                 | Early stopping, validation loss tracking       |

---

## 📌 Key Learnings

- Successfully learned how to **condition UNet architectures** with textual inputs.
- Explored image-to-image generation tasks involving **semantic control via embeddings**.
- Practiced real-world ML ops via **WandB logging, experiment tracking**, and deployment setup.
- Reinforced understanding of **model generalization**, dataset preprocessing, and augmentation strategies.

---

## 📤 Deliverables

- ✅ UNet Model & Training Script (`model.py`, `train.py`)
- ✅ Inference Notebook ('test.ipynb')
- ✅ WandB Link: [colored-polygon-unet](https://wandb.ai/salonisahal15-university-of-kalyani/colored-polygon-unet/runs/yz0igigf)
- ✅ This `README.md`

---

## 🚀 Future Work

- Extend to more complex shapes and multi-color fill patterns.
- Add attention to color embedding influence.
- Try generative diffusion models for higher-quality fills.

---

*Prepared by Saloni Sahal *
