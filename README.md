# Desert Semantic Segmentation: Synthetic to Real World

**Duality AI Offroad Semantic Segmentation Submission README**

## 📌 Project Overview
This project implements a semantic segmentation model trained on synthetic desert data provided by Duality AI. The objective is to segment terrain classes at the pixel level and evaluate generalization performance on unseen desert environments. 

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
