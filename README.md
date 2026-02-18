# Desert Semantic Segmentation: Synthetic to Real World

**Duality AI Offroad Semantic Segmentation Submission README**

## Project Overview
This project implements a semantic segmentation model trained on synthetic desert data provided by Duality AI. The objective is to segment terrain classes at the pixel level and evaluate generalization performance on unseen desert environments. 

The pre trained model here is not included as it exeeds upload limit on github
but the model can be downloaded from the provided Gdrive link in the *Model_link.txt* file. 
the user can also provide their own dataset for training or use a separate pretrained model

**Primary evaluation metric:** Mean Intersection over Union (IoU).

---

## ⚙️ Environment & Dependencies

### System Requirements:
* **OS:** Windows 
* **IDE:** Visual Studio Code
* **Language:** Python 3.x
* **Hardware:** Dedicated GPU (CUDA Supported)

### Environment Setup

**1. Create virtual environment:**
```bash
python -m venv myenv
```
**2. Activate the Environment:**
```bash
.\myenv\Scripts\activate
```
**3. Install GPU Dependencies (PyTorch)**
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
```
**4. Install Project Libraries**
```bash
pip install streamlit opencv-python pandas numpy pillow
```
**How to Run**
```bash
streamlit run front_v2.py
```
## Methodology

### 1. Model Architecture
* **Network:** DeepLabV3 with a **ResNet50** backbone.
* **Reasoning:** DeepLabV3 utilizes Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale contextual information. This is highly effective for segmenting varied desert terrains, helping the model understand both vast landscapes and small, intricate objects like dry bushes.

### 2. Data Preprocessing
* **Resolution:** All input images and ground truth masks are standardized to `256x256` pixels for optimal memory utilization and faster computation.
* **Class Mapping:** The raw dataset masks contained specific arbitrary pixel values (e.g., 100, 200, 7100) which were systematically mapped to 10 sequential classes (0-9). 
* **Ignore Index:** Unclassified or border pixels were remapped to `255` and explicitly excluded from the loss calculation (`ignore_index=255`) to prevent skewed training results.

### 3. Training Strategy
* **Hardware:** Trained leveraging an NVIDIA RTX 4060 GPU (8GB VRAM) for accelerated processing.
* **Hyperparameters:** Batch Size = 8, Total Epochs = 40.
* **Loss Function:** `CrossEntropyLoss`.
* **Optimizer:** `Adam` optimizer with an initial learning rate of `0.0002` for fast, adaptive, and stable convergence.
* **Learning Rate Scheduler:** `StepLR` (step_size=20, gamma=0.1). The learning rate is actively decayed by 90% after the 20th epoch. This technique allows the model to initially learn broad features (like sky and ground) and later fine-tune its weights to detect smaller, difficult classes with higher precision.


**Output samples**

<img width="1920" height="1080" alt="Screenshot 2026-02-18 184340" src="https://github.com/user-attachments/assets/d8a08d69-04f3-4d12-bd1e-c1b00fcca5b8" />
<img width="1920" height="1080" alt="Screenshot 2026-02-18 184349" src="https://github.com/user-attachments/assets/896837a7-b040-4dc1-b2aa-dba85511e62a" />

*The resultant meaan IOU was: 0.2537*
*with a Pixel accuracy of : 60.68%*
*and a Inference Time of : 241.58 ms*
*also a highest mAP50 score of : 0.25*
