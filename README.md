# Desert Semantic Segmentation: Synthetic to Real World

**Duality AI Offroad Semantic Segmentation Submission README**

## Project Overview
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

<img width="1920" height="1080" alt="Screenshot 2026-02-18 174745" src="https://github.com/user-attachments/assets/03f57104-e7cc-444b-83a4-921bef890441" />
<img width="1920" height="1080" alt="Screenshot 2026-02-18 174753" src="https://github.com/user-attachments/assets/0c40c5bc-44d0-4be4-bb68-341949fdfdc5" />

The resultant meaan IOU was: 0.2537
with a Pixel accuracy of : 60.68%
and a Inference Time of : 241.58 ms
