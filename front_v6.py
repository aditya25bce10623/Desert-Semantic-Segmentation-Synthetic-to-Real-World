import streamlit as st
import numpy as np
from PIL import Image
import time
import pandas as pd
import cv2
import os
import torch
import torchvision.transforms as T
import torch.nn.functional as F

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    if os.path.exists("model.pth"):
        m = torch.load("model.pth", map_location="cpu", weights_only=False)
        m.eval()
        return m
    return None

m = load_model()

cls = [
    'Sky', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape'
]

cols = [
    [135, 206, 235], [34, 139, 34], [0, 255, 0], [189, 183, 107], [139, 69, 19],
    [160, 82, 45], [255, 20, 147], [160, 82, 45], [128, 128, 128], [210, 180, 140]
]

st.title("Desert Semantic Segmentation Dashboard")

up_img = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png'])
up_gt = st.sidebar.file_uploader("Upload Ground Truth", type=['png'])

op = st.sidebar.slider("Opacity", 0.0, 1.0, 0.5)

sel_cls = []
for i in range(len(cls)):
    if st.sidebar.checkbox(cls[i], value=True):
        sel_cls.append(i)

if up_img:
    img_p = Image.open(up_img).convert("RGB")
    img_a = np.array(img_p)
    
    t1 = time.time()
    
    h = img_a.shape[0]
    w = img_a.shape[1]
    
    if m is not None:
        tr = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        in_t = tr(img_p).unsqueeze(0)
        
        with torch.no_grad():
            out = m(in_t)["out"]
            probs = F.softmax(out[0], dim=0)
            conf_m = torch.max(probs, dim=0)[0].cpu().numpy()
            
        p_t = torch.argmax(out, dim=1).squeeze(0)
        p_m_s = p_t.byte().cpu().numpy()
        p_m = cv2.resize(p_m_s, (w, h), interpolation=cv2.INTER_NEAREST)
        c_m_r = cv2.resize(conf_m, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        p_m = np.random.randint(0, len(cls), (h, w))
        c_m_r = np.random.uniform(0.5, 1.0, (h, w))
    
    t2 = time.time()
    inf_t = round((t2 - t1) * 1000, 2)
    
    ov = img_a.copy()
    for i in range(len(cols)):
        if i in sel_cls:
            mask_i = p_m == i
            ov[mask_i] = cols[i]
            
    disp_img = cv2.addWeighted(ov, op, img_a, 1 - op, 0)
    
    if m is None:
        st.warning("model.pth file nahi mili!")
    
    st.write(f"Inference Time: {inf_t} ms")
    
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_p, caption="Original Image", use_container_width=True)
    with c2:
        st.image(disp_img, caption="Predicted Segmentation", use_container_width=True)
        
    gt_m = None
    if up_gt:
        gt_p = Image.open(up_gt)
        gt_r = gt_p.resize((w, h), Image.NEAREST)
        gt_a = np.array(gt_r)
        
        gt_m = np.full(gt_a.shape, 255, dtype=np.int64)
        gt_m[gt_a == 100] = 0
        gt_m[gt_a == 200] = 1
        gt_m[gt_a == 300] = 2
        gt_m[gt_a == 500] = 3
        gt_m[gt_a == 550] = 4
        gt_m[gt_a == 600] = 5
        gt_m[gt_a == 700] = 6
        gt_m[gt_a == 800] = 7
        gt_m[gt_a == 7100] = 8
        gt_m[gt_a == 10000] = 9
        
        gt_v = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(len(cols)):
            gt_v[gt_m == i] = cols[i]
            
        err = np.where((p_m != gt_m) & (gt_m != 255), 255, 0).astype(np.uint8)
        
        c3, c4 = st.columns(2)
        with c3:
            st.image(gt_v, caption="Ground Truth Mask", use_container_width=True)
        with c4:
            st.image(err, caption="Error Map", use_container_width=True)

    st.subheader("Metrics Table")
    
    m_data = []
    for i in range(len(cls)):
        p_i = (p_m == i)
        
        if np.sum(p_i) > 0:
            conf_val = round(np.mean(c_m_r[p_i]) * 100, 1)
            conf_s = f"{conf_val}%"
        else:
            conf_s = "N/A"
            
        if gt_m is not None:
            g_i = (gt_m == i)
            inter = np.logical_and(g_i, p_i).sum()
            union = np.logical_or(g_i, p_i).sum()
            
            if union > 0:
                iou_v = round(inter / union, 4)
            else:
                iou_v = "N/A"
        else:
            iou_v = "N/A (Upload GT)"
            
        m_data.append([cls[i], iou_v, conf_s])
        
    m_df = pd.DataFrame(m_data, columns=["Class", "IoU Score", "Confidence"])
    
    st.table(m_df)