# Adapted from https://github.com/luca-medeiros/lang-segment-anything/blob/main/example_notebook/getting_started_with_lang_sam.ipynb
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from lang_sam import LangSAM
import os
os.environ["HF_HOME"] = "/mnt/files/zocket/huggingface/"
import torch
#from diffusers import AutoPipelineForInpainting,AutoPipelineForImage2Image
#from diffusers.utils import load_image, make_image_grid
import numpy as  np
from task2_utils import *

def main():

    parser = argparse.ArgumentParser(description='Fill the class object pixels with red.')
    parser.add_argument('--image', help='Path to the input image containing the target object',default="image_inputs/test.jpg")
    parser.add_argument('--class_name', help='Class name of the target object to be masked',default="person")
    parser.add_argument('--output', help='Path to save the output image with the red mask',default="image_outputs/test.jpg")

    args = parser.parse_args()


    model = LangSAM(ckpt_path="model-weights/sam_vit_h_4b8939.pth")
    image_pil = Image.open(args.image)
    masks, boxes, phrases, logits = model.predict(image_pil.convert("RGB"), args.class_name)
    mask_pil=masks.numpy()[0].astype(np.int8)
    
    mask_pil=Image.fromarray(mask_pil*255)
    plt.imsave(args.output,np.array(mask_pil))

    if len(masks) == 0:
            print(f"No objects of the '{args.class_name}' prompt detected in the image.")
    else:
            # Convert masks to numpy arrays
            masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

            # Display the original image and masks side by side
            display_image_with_masks(image_pil, masks_np)

            # Display the image with bounding boxes and confidence scores
            display_image_with_boxes(image_pil, boxes, logits)

            # Save the masks
            for i, mask_np in enumerate(masks_np):
                mask_path = f"image_outputs/image_mask_{i+1}.png"
                save_mask(mask_np, mask_path)

            # Print the bounding boxes, phrases, and logits
            print_bounding_boxes(boxes)
            print_detected_phrases(phrases)
            print_logits(logits)




    
if __name__ == "__main__":

    main()