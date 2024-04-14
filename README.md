# AI-Enhanced Product Photoshoot Visuals and Filter

<p float="left" align="middle">
  <img src="screenshots/00007-778790675.png" width="500" />  
</p>


An experiment to generate highly accurate product photography using Low Rank Adapation (LORA) on Stable Diffusion.
  
[Click Here](#generated-images) to scroll to generated images from finetuned Stable Diffusion.  
[Click Here](#sample-code-run-results) to look at the results of the filter to detect the presence of an product.


# Problem Statement

In this task, I want to accomplish the three key objectives
1. **Generative AI for Visuals**: Design an AI Model to generate product photoshoot visuals
2. **Product Recognition Filter**: AI based filter to identify and isolate specific products in a given image. If the object is present, enhance the visual appearence while preserving other parts of the image.
3. **Exclusion of Non-Relevant Images**: If none of the products specified are present, skip the image without any additional processing techniques.


# Overview of the Approach

The approach I have taken to solve this is based on three steps

-  **Step 1** - To generate accurate product photoshoots, I fine tune the **Stable Diffusion 1.5** model from RunwayML [(runwayml/stable-diffusion-v1-5)](https://huggingface.co/runwayml/stable-diffusion-v1-5) using LoRA on selected images of product photoshoot visuals.  


- **Step 2** - Once the visuals are generated I use [Segment Anything](https://github.com/facebookresearch/segment-anything) from Facebook and [CLIP](https://github.com/openai/CLIP) from OpenAI to classify the segmentation masks using text prompts. If the given product that is provided with the text prompt is found, the mask of the found object is used for further processing and optimising the chosen product.

- **Step 3** -   To enhance the visual appearence of the project or to even modify the attributes of the product, I use Image2Image Inpainting along with [ControlNet](https://github.com/lllyasviel/ControlNet) to more finely control the product while preserving the scene. This is made possible by the product mask generated from the previous step.

Each of these steps are explained in more detail in the next sections.

# Explanation of the Process

## Step 1

Even though stable diffusion is quite powerful to generate realistic  images, it can be even more customised to generate an particular art style. I see product photography as also a form of art style that can be learnt by the model. 

### Training Methodology

Due to constraints with GPU capacity and time, fine-tuning the all the parameters and weights of the Stable Diffusion model is quite tedious. Therefore, after initial literature search, I used the [ Low-Rank Adaptation of Large Language Models (LoRA)](https://arxiv.org/abs/2106.09685) technique proposed by Edward J et. al.

Using this technique, it is possible to finetune LLMs for downstream task by inserting small number of additional weights into the model and training them. This makes the fine tuning stable diffusion much faster to experiment and run on consumer grade GPUs.

### Training Data

For finetuning, I sourced images from [Unsplash](https://unsplash.com/). Unsplash provides free high quality images. The names of images along with the photographers are cited in this [file](lora_training_images/image_list.txt).


<p float="left" align="middle">
  <img src="lora_training_images/13.jpg" width="100" />
  <img src="lora_training_images/22.jpg" width="100" /> 
  <img src="lora_training_images/33.jpg" width="100" />
</p>

### Generating Text Prompts

Once the images are selected, the text prompts also need to be generated for finetuning stable. For generating the prompts, I used (BLIP: Bootstrapping Language-Image Pre-training)(https://arxiv.org/abs/2201.12086) from Salesforce that has the capabilities of image Captioning.

I use the framework provided by [kohya-colab](https://github.com/hollowstrawberry/kohya-colab) to generate these text prompts. The [Rahul Dataset Maker](notebooks/Rahul_Dataset_Maker.ipynb) notebook is run on Google Colab service for the sake of quick experiments and faster hardware. Once the prompts are generated, these are stored as a text file with the same name as the image file to be later used for training. The set of training images and text prompts are available at [lora_training_images](lora_training_images).

### Activation Tag

In addition to the tag generated from BLIP, another activation tag called `ppzocketv2` is added to all the training images. During inference, when the same activation tag is used in the text prompt, this will help the stable diffusion model to generate images more closely to the training images.

### Finetuning the Stable Diffusion Model

Once again I use the [Rahul Lora Trainer.ipynb](notebooks/Rahul_Lora_Trainer.ipynb) adapted from [kohya-colab](https://github.com/hollowstrawberry/kohya-colab) and is again run on Google Colab. 

#### Training Parameters

- Total Number of Images: 65  
- Number of Repeats: 6  
- Batch Size: 2  
- Number of Epochs: 10   
- Steps per Epoch - 65*6/2 - 195 Steps
- Total Training Steps - 1950

The trained LoRA checkpoint is available [here](lora_checkpoint/ppzocketv2-10.safetensors).

### Generating Product Visuals from LoRA Checkpoint

Once we have the LoRA checkpoint, the next step is to perform inference on the trained model to generate product visuals. For this purpose, I use
[Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that provides an GUI based inference platform for running stable diffusion models.

For prompts, I use the following template 

    ppzocketv2, "class", stylish, studio photography, product photography, ultra realistic, <lora:ppzocketv2-10:1>

`ppzocketv2` acts as the LoRA activation tag, `"class"` denotes the product class for which we need to generate visual. `<lora:ppzocketv2-10:1>` instructs the Web UI to use the finetuning checkpoint along with the base model. 

Sample Screenshot of the WebUI

<p float="left" align="middle">
  <img src="screenshots/web_ui_generate.png" width="1000" />
</p>

### Generated Images
The image generated for different product visuals along with the prompt is located at [task1_outputs](task1_outputs). The thumbnails of the generated image is here:

<p float="left" align="middle">
  <img src="task1_outputs/car/00050-65.png" width="100" />
  <img src="task1_outputs/car/00077-68.png" width="100" /> 
  <img src="task1_outputs/car/00078-42.png" width="100" />
</p>
<p float="left" align="middle">
  <img src="task1_outputs/handbag/00014-42.png" width="100" />
  <img src="task1_outputs/handbag/00015-43.png" width="100" /> 
  <img src="task1_outputs/handbag/00016-44.png" width="100" />
  <img src="task1_outputs/handbag/00017-45.png" width="100" />
</p>
<p float="left" align="middle">
  <img src="task1_outputs/leather_shoes/00044-44.png" width="100" />
  <img src="task1_outputs/leather_shoes/00046-65.png" width="100" />
</p>
<p float="left" align="middle">
  <img src="task1_outputs/sport_shoes/00027-43.png" width="100" />
  <img src="task1_outputs/sport_shoes/00029-45.png" width="100" />
  <img src="task1_outputs/sport_shoes/00036-44.png" width="100" />
</p>
<p float="left" align="middle">
  <img src="task1_outputs/wristwatch/00082-42.png" width="100" />
  <img src="task1_outputs/wristwatch/00088-598.png" width="100" />
  <img src="task1_outputs/wristwatch/00091-597.png" width="100" />
</p>

## Step 2

In this step, the task ahead is to find if any of the products are present in the image. Additionally, for improving the product appearence, the mask of where the product is located also needs to be stored. Once we have the binary mask of the product, we can use it as guide for inpainting in the next step.

The Segment Anything model is highly accurate for identifying foreground object but however it does not classify between the different foreground objects. For the task ahead, the position as well as the class that the foreground object belongs to is also required. For classifying the proposed bounding boxes, I used CLIP that demonstrates strong zero-shot capabilities. Therefore, for each bounding box, I check if any of them matches with prompts generated from the class names and classify them.

I rescrape the script from [segment-anything-with-clip](https://github.com/Curt-Park/segment-anything-with-clip) and adapt it for the task of multi-class classification. The code is available at [image_filter.py](image_filter.py) and [image_filter_utils.py](image_filter_utils.py)


### Sample Code Run

    python image_filter.py

The input images used for this test are located at [filter_image_inputs](filter_image_inputs) and the outputs generated from the script are saved at [filter_image_outputs](filter_image_outputs). **All the input images were generated from the finetuned model in Step 1.** 

The product class can be described in the configration file at [config.yaml](conf/config.yaml). For the purposes of the experiment the following product classes are used  `["hand bag", "shoe","car","watch or watch dial"]`.


### Sample Code Run Results

Here is the screenshot from the console.

<p float="left" align="middle">
  <img src="screenshots/filter_output.png" width="500" />
</p>

As is it seen from the screenshot, the program outputs if there is any of the product class in the image and the number of such regions found. For the final image, which is that of moon, we can see that none of the product class are present.

In the visualisation below, each row represents a separate input image. The first image of each row represents the input image and the following images are the mask generated on the identified object. For the last image, since no products in the given list were identified, no masks are generated.


<p float="left" align="middle">
  <img src="filter_image_inputs/handbag.png" width="100" />
  <img src="filter_image_outputs/handbag/hand bag/mask_0.png" width="100" /> 
</p>

<p float="left" align="middle">
  <img src="filter_image_inputs/shoes.png" width="100" />
  <img src="filter_image_outputs/shoes/shoe/mask_0.png" width="100" /> 
</p>
<p float="left" align="middle">
  <img src="filter_image_inputs/car.png" width="100" />
  <img src="filter_image_outputs/car/car/mask_0.png" width="100" /> 
</p>

<p float="left" align="middle">
  <img src="filter_image_inputs/watch.png" width="100" />
  <img src="filter_image_outputs/watch/watch or watch dial/mask_0.png" width="100" /> 
  <img src="filter_image_outputs/watch/watch or watch dial/mask_1.png" width="100" />
  <img src="filter_image_outputs/watch/watch or watch dial/mask_2.png" width="100" />
  <img src="filter_image_outputs/watch/watch or watch dial/mask_3.png" width="100" />
  </p>

<p float="left" align="middle">
  <img src="filter_image_inputs/unknown.png" width="100" />
</p>

## Step 3

For the final step, I once again use the [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for enhancing the product visuals. Additionally, I also use ControlNet to finetune the product details. The images and prompts are available in the folder [controlnet_images](controlnet_images).

I use the main image for inpainting through the Image-to-Image generation pipeline. I specify the region to inpaint using the product mask generated in the last step. For ControlNet, I use `control_v11f1p_sd15_depth` model from [lllyasviel/control_v11f1p_sd15_depth](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth).

The idea behind using ControlNet is to preserve the position and background details of image and only vary the appearence in the inpainted region. By using a ControlNet model based on Depth, the depth map in the source and generated image will be relatively similar. There are also lot more ControlNet models that can be experimented with.


<p float="left" align="middle">
  <img src="controlnet_images/car/car.png" width="100" />
  <img src="controlnet_images/car/mask_0.png" width="100" /> 
  <img src="controlnet_images/car/00084-3874773259.png" width="100" />
  <img src="controlnet_images/car/00093-3921352636.png" width="100" />
</p>

  <p float="left" align="middle">
  <img src="controlnet_images/handbag/handbag.png" width="100" />
  <img src="controlnet_images/handbag/mask_0.png" width="100" /> 
  <img src="controlnet_images/handbag/00052-3434634984.png" width="100" />
  <img src="controlnet_images/handbag/00053-474029979.png" width="100" />
  </p>

  <p float="left" align="middle">
  <img src="controlnet_images/watch/watch.png" width="100" />
  <img src="controlnet_images/watch/mask_2.png" width="100" /> 
  <img src="controlnet_images/watch/00056-534882995.png" width="100" />

  </p>
