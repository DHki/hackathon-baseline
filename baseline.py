import torch
import torchvision
import json
import numpy as np
import cv2

from torchvision.transforms import functional
from PIL import Image

# 8 COLORS for 8 classes
COLORS = [
    (128, 128, 128) # GRAY
    (255, 0, 0), # RED
    (0, 255, 0), # GREEN
    (0, 0, 255), # BLUE
    (255, 255, 0), # YELLOW
    (0, 255, 255), # CYAN
    (255, 0, 255), # MAGENTA
    (255, 255, 255) # WHITE
]

def load_image(path):
    image = Image.open(path).convert("RGB")
    image_tensor = functional.to_tensor(image).unsqueeze(0)
    return image, image_tensor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## get mask RCNN Model and set for interference ##
    mask_RCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    mask_RCNN.eval()

    mask_RCNN.to(device)
    
    with open('./data/annotations/val.json', 'r', encoding='utf-8') as file:
        annota = json.load(file)
    
    cnt = 0
    
    for info in annota['images']:
        if cnt == 5:
            del img_tensor
            torch.cuda.empty_cache()
            cnt = 0
        
        file_name = info['file_name']
    
        img, img_tensor = load_image(f'./data/val/{file_name}')
        img_tensor = img_tensor.to(device)
    
    
        ## inference ##
        prediction = mask_RCNN(img_tensor)
        
        masks = prediction[0]['masks'].squeeze().detach().cpu().numpy()
        scores = prediction[0]['scores'].detach().cpu().numpy()
        labels = prediction[0]['labels'].detach().cpu().numpy()
    
        img_mask = np.array(img)
    
        # for mask, score in zip(masks, scores):
        #     if score > 0.2:
        #         color = np.array([255, 0, 0], dtype=np.uint8)
        #         colored_mask = np.zeros_like(img_mask)
                
        #         for i in range(3):
        #             colored_mask[:, :, i] = color[i] * mask
        #         img_mask = cv2.addWeighted(img_mask, 1, colored_mask, 1.0, 0)

        for mask, score, label in zip(masks, scores, labels):
            label = label % 8 # 8 classes
            
            if score > 0.2:
                color = COLORS[label]
                colored_mask = np.zeros_like(img_mask)
                
                for i in range(3):
                    colored_mask[:, :, i] = color[i] * mask
                img_mask = cv2.addWeighted(img_mask, 1, colored_mask, 1.0, 0)
        
        img_store = Image.fromarray(img_mask)
        img_store.save(f'./results/{file_name}')
        
        print(f"saved {file_name}")
        cnt += 1
    


if __name__ == "__main__":
    main()