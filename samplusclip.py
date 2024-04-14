import os
import urllib
from functools import lru_cache
from random import randint
from typing import Any, Callable, Dict, List, Tuple

import clip
import cv2
import numpy as np
import PIL
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from samplusclip_utils import *



@hydra.main(version_base=None,config_path="conf", config_name="config.yaml")
def main(cfg : DictConfig):

    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry["vit_b"](checkpoint= "model-weights/sam_vit_b_01ec64.pth").to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    

    input_images=os.listdir(cfg.image.input_folder)
    for each_input_image in input_images:
        image_path =os.path.join(cfg.image.input_folder,each_input_image)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = adjust_image_size(image)
        masks = mask_generator.generate(image)
        #print(masks)
         
        masks = filter_masks(
        image,
        masks,
        predicted_iou_threshold=0.9,
        stability_score_threshold=0.8,
        query="handbag",
        clip_threshold=0.85,
    )
        image = draw_masks(image, masks)
        image = PIL.Image.fromarray(image)
        image.show()

    
if __name__ == "__main__":

    main()