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


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    # sam = sam_model_registry["vit_h"](checkpoint= "model-weights/sam_vit_h_4b8939.pth").to(device)

    sam = sam_model_registry["vit_b"](
        checkpoint="model-weights/sam_vit_b_01ec64.pth"
    ).to(device)

    input_images = os.listdir(cfg.image.input_folder)
    for each_input_image in input_images:
        image_path = os.path.join(cfg.image.input_folder, each_input_image)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_generator = SamAutomaticMaskGenerator(
            sam, min_mask_region_area=0.25 * image.shape[0] * image.shape[1]
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = adjust_image_size(image)
        masks = mask_generator.generate(image)
        output_dict = {}
        output_mask = []

        # dummy_image = draw_masks(image, masks)
        # dummy_image = PIL.Image.fromarray(dummy_image)
        # dummy_image.show()
        for each_class in cfg.product_names:

            query_dict, masks = filter_masks(
                image,
                masks,
                predicted_iou_threshold=0.9,
                stability_score_threshold=0.8,
                query=cfg.product_names,
                clip_threshold=0.85,
            )
            if len(masks):
                # output_dict[query] = masks
                output_mask.extend(masks)

        image = draw_masks(image, output_mask)
        print(each_input_image)
        for key, value in query_dict.items():
            print(each_input_image, key, len(value))
        image = PIL.Image.fromarray(image)
        image.show()
        pass


if __name__ == "__main__":

    main()
