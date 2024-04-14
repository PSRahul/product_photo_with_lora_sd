# AI-Enhanced Product Photoshoot Visuals and Filter

An experiment to generate highly accurate product photography using Low Rank Adapation (LORA) on Stable Diffusion

# Problem Statement

In this task, I want to accomplish the three key objectives
1. **Generative AI for Visuals**: Design an AI Model to generate product photoshoot visuals
2. **Product Recognition Filter**: AI based filter to identify and isolate specific products in a given image. If the object is present, enhance the visual appearence while preserving other parts of the image.
3. **Exclusion of Non-Relevant Images**: If none of the products specified are present, skip the image without any additional processing techniques.

# Approach

The approach I have taken to solve this is based on three steps

**Step 1** - To generate accurate product photoshoots, I fine tune the **Stable Diffusion 1.5** model from RunwayML [(runwayml/stable-diffusion-v1-5)](https://huggingface.co/runwayml/stable-diffusion-v1-5) on selected images of product photoshoot visuals.  


**Step 2** - ** Once the visuals are generated I use [Language Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything) that builds on [Segment Anything](https://github.com/facebookresearch/segment-anything)from Facebook and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) from IDEA Research to guide image segmentation using text prompts.  

**Step 3** -  If the given product that is provided with the text prompt is found, the image is used for further processing and optimising the chosen product. To accomplish this I use [ControlNet](https://github.com/lllyasviel/ControlNet) to more finely control the product while preserving the scene.

Each of these steps are explained in more detail in the next sections.

# Solutions

Task 1 Prompt - ppzocketv2, "class", stylish, studio photography, product photography, ultra realistic, <lora:ppzocketv2-10:1>

