import os

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from image_filter_utils import filter_masks, adjust_image_size


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry["vit_b"](
        checkpoint="model-weights/sam_vit_b_01ec64.pth"
    ).to(device)

    input_images = os.listdir(cfg.input_folder)
    print("List of Input Images \n", input_images)
    for each_input_image in input_images:
        print(f"Processing Input Image - {each_input_image}")

        image_path = os.path.join(cfg.input_folder, each_input_image)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_generator = SamAutomaticMaskGenerator(
            sam, min_mask_region_area=0.25 * image.shape[0] * image.shape[1]
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = adjust_image_size(image)
        masks = mask_generator.generate(image)

        query_dict = filter_masks(
            image,
            masks,
            predicted_iou_threshold=0.9,
            stability_score_threshold=0.8,
            query=cfg.product_names,
            clip_threshold=0.85,
        )

        for key, value in query_dict.items():
            print(f"Found {len(value)} instances of {key} in the image")
            save_path = os.path.join(
                cfg.output_folder, each_input_image.split(".")[0], key
            )
            os.makedirs(save_path, exist_ok=True)
            for idx, each_mask in enumerate(value):
                plt.imsave(
                    os.path.join(save_path, "mask_" + str(idx) + ".png"),
                    each_mask["segmentation"],
                )

        if len(query_dict.keys()) == 0:
            print(f"No instances of {cfg.product_names} were found in the image")


if __name__ == "__main__":

    main()
