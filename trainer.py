import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

class DesertDataset(Dataset):
    def __init__(self, img_path, mask_path):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_names = os.listdir(img_path)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        img_file = os.path.join(self.img_path, self.img_names[i])
        mask_file = os.path.join(self.mask_path, self.img_names[i])
        
        img = Image.open(img_file).convert("RGB")
        img = img.resize((256, 256))
        img_tensor = ToTensor()(img)
        
        mask = Image.open(mask_file)
        mask = mask.resize((256, 256), Image.NEAREST)
        mask_arr = np.array(mask)
        
        new_mask = np.full(mask_arr.shape, 255, dtype=np.int64)
        new_mask[mask_arr == 100] = 0
        new_mask[mask_arr == 200] = 1
        new_mask[mask_arr == 300] = 2
        new_mask[mask_arr == 500] = 3
        new_mask[mask_arr == 550] = 4
        new_mask[mask_arr == 600] = 5
        new_mask[mask_arr == 700] = 6
        new_mask[mask_arr == 800] = 7
        new_mask[mask_arr == 7100] = 8
        new_mask[mask_arr == 10000] = 9
        
        mask_tensor = torch.tensor(new_mask, dtype=torch.long)
        
        return img_tensor, mask_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(weights=None, num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=255)
opt = torch.optim.Adam(model.parameters(), lr=0.0002)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)

train_imgs = r"D:\statathon\Offroad_Segmentation_Training_Dataset\train\color_images"
train_masks = r"D:\statathon\Offroad_Segmentation_Training_Dataset\train\Segmentation"
epochs = 40

dataset = DesertDataset(train_imgs, train_masks)
loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
total_batches = len(loader)

for e in range(epochs):
    model.train()
    total_loss = 0
    batch_num = 1
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        opt.zero_grad()
        out = model(x)["out"]
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        
        print("Epoch:", e+1, "Batch:", batch_num, "/", total_batches, "Loss:", round(loss.item(), 4))
        batch_num += 1
        
    sched.step()
    curr_lr = sched.get_last_lr()[0]
    print("Epoch", e+1, "Completed! Total Loss:", round(total_loss, 4), "LR:", curr_lr)
    
torch.save(model, r"D:\statathon\model.pth")
