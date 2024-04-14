# Adapted from https://github.com/luca-medeiros/lang-segment-anything/blob/main/example_notebook/getting_started_with_lang_sam.ipynb

import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from lang_sam import LangSAM
import os
os.environ["HF_HOME"] = "/mnt/files/zocket/huggingface/"
import torch
import numpy as  np
from task2_utils import *


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg : DictConfig):

    print(OmegaConf.to_yaml(cfg))

    model = LangSAM(ckpt_path="model-weights/sam_vit_h_4b8939.pth")

    input_images=os.listdir(cfg.image.input_folder)
    for each_input_image in input_images:
        image_pil = Image.open(os.path.join(cfg.image.input_folder,each_input_image)).convert("RGB")
                
        for each_class in cfg.product_names:
                masks, boxes, phrases, logits = model.predict(image_pil, f"only detect {each_class}")
                print(boxes,phrases)
                mask_pil=masks.numpy()[0].astype(np.int8)
                print(each_input_image,each_class)
                mask_pil=Image.fromarray(mask_pil*255)
                mask_pil.show()

    


    
if __name__ == "__main__":

    main()