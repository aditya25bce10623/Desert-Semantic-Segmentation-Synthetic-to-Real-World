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
**Output samples**
<img width="1920" height="1020" alt="Screenshot 2026-02-18 174012" src="https://github.com/user-attachments/assets/834aad76-b55c-4e71-86f9-1a12be448127" />

<img width="1920" height="1020" alt="Screenshot 2026-02-18 173942" src="https://github.com/user-attachments/assets/aa82c80f-f86a-41d3-8358-01a8c0b94f8e" />
