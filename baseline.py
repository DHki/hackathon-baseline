import torch
import torchvision
import json
import numpy as np
import cv2

from torchvision.transforms import functional
from PIL import Image

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

        img_mask = np.zeros(img.size, dtype=np.uint8)
        for mask, score, label in zip(masks, scores, labels):
            label = (label % 7) + 1 # 7 classes
            
            if score > 0.2:
                mask = mask.squeeze()
                
                mask_resized = np.transpose(mask, (1, 0))
                img_mask[mask_resized > 0.2] = label
        
        img_mask = np.transpose(img_mask, (1, 0))
        np.savetxt(f"./results/ans_{file_name.split('.')[0]}.csv", img_mask, delimiter=", ")
        
        print(f"saved {file_name}")
        cnt += 1
    


if __name__ == "__main__":
    main()