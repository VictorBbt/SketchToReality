# SketchToReality: Enhancing AI's Understanding and Creation of art

![Sketchy vs Diffusion](https://github.com/VictorBbt/SketchToReality/blob/main/img/intro.png "Intro Image")

This is the final project done for the course Advanced Computer Vision given by Vicky Kalogeiton at Ecole Polytechnique. This project aims to assess how models deal with stylistic differences between images such as realistic images ad simple sketches. 

More precisely, we finetune diffusion models on the [Sketchy Database](https://faculty.cc.gatech.edu/~hays/tmp/sketchy-database.pdf) to try to incorporate the sketch style within diffusion models. To get acquainted with the database, you ca either run locally (resp. on Colab) the notebook **data_visualization.ipynb** (resp **data_visualization.ipynb**).
To achieve our goal, we investigated two approaches:

- **Text-in-the-middle approach:** use Image Captioning Models we first generate captions with pretrained models such as [Llava](https://huggingface.co/docs/transformers/model_doc/llava) and [Blip2](https://huggingface.co/docs/transformers/model_doc/blip-2).
- **Direct Method:** directly finetune Diffusion Models using [DreamBooth](https://dreambooth.github.io) or [Textual Inversion](https://textual-inversion.github.io).

## Text-in-the-middle approach

Although textual description of sketches can be whether to simple or clumsy (explain verbally the pose ad the different features of the sketches), we thought that is was possible to describe sketches with enough precision using Image Captioning Models. These captions are then used to generate images with [StableDiffusions](https://huggingface.co/spaces/stabilityai/stable-diffusion). 

First, we identify and try ten different prompts (we used ChatGPT to get ideas) that could help us get precise captions. We then compare the captions produced by two different models, namely [Llava](https://huggingface.co/docs/transformers/model_doc/llava) and [Blip2](https://huggingface.co/docs/transformers/model_doc/blip-2) qualitatively and quantitatively (length of the answers, accuracy). The different prompts and answers provided by Llava are available in the **Inference_with_LLava_for_multimodal_generation.ipynb**. In **Inference_with_BLIP_2_(int8).ipynb**, we only try the 3 best prompts found with Llava to generate captions.
Then, we generate images with StableDiffusions1.5, and assess the quality of the generated images with the **Fréchet Inception Distance**, using [this implementation](https://github.com/mseitzer/pytorch-fid).

## Direct Approach

In this part, we use recent models to incorporate specific subjects in diffusion models. The goal is to convey, with few sketches, an idea of what is a sketch. The generated images can be found in *mosaics*.

The most recent model is [DreamBooth](https://dreambooth.github.io), that consists in attaching a subject (preset in 1~5 images) to an unknown word of the Diffusion Model's vocabulary. This methods finetunes all the weights of the model. We used DreamBooth with different sketches, whether of the same class (cat, dog) or with a sequential training with different classes. The training code and inference examples can be found in the notebooks **dreambooth.ipynb** and **dreambooth_sequential.ipynb**. 
 We also try to finetune a diffusion model with [Textual Inversion](https://textual-inversion.github.io) in **textual_inversio.ipynb**

Finally, we generate images for the same 100 images as in the text-in-the-middle approach in **Inference_with_Dreambooth.ipynb**, and assess the quality of the generation.

## Running the code

Running the notebook should be seamless in any cuda environment, as all the required installations are done within the notebooks. To build the *train, test* and *validation* splits for the Sketchy database, run the codes provided in **utils**. To do so, you should already have downloaded in your local repository the original dataset.

### So what is art ?

If these questions about the nature of art and the notion of creation interest you, you can re d (in French) the book of Alban Leveau-Vallier *IA: l'intuition à l'épreuve des algorithmes*. Now, we leave you with these beautiful (real or generated ?) butterflies 🦋
![butterflies](https://github.com/VictorBbt/SketchToReality/blob/main/img/butterfly.png "Butterfly")

