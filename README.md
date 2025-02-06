# **imagination_to_real** by SmilingRobo 

### [🌐 SmilingRobo](https://www.smilingrobo.com) | [📝 Paper](https://arxiv.org/abs/2411.00083) | [Open-Source Sprint](https://opensource.smilingrobo.com/projects/imagination-to-real/)

**imagination-to-real** Train your robot to do whatever you want using Generative AI

#### Description 
imagination-to-real empowers robotics developers by bridging the gap between generative AI and classical physics simulators. Our library prepares realistic, diverse, and geometrically accurate visual data from generative models. This data enables robots to learn complex and highly dynamic tasks, such as parkour, without requiring depth sensors.

🚀 What It Does:

⚪ Integrates generative models with simulators to create rich, synthetic datasets.<br>
⚪ Ensures temporal consistency with tools like Dreams In Motion (DIM).<br>
⚪ Offers compatibility with MuJoCo environments for seamless data preparation.<br>

🛠️ How to Use:

⚪ Use Image_Maker for text-to-image generation tailored to your simulation needs.<br>
⚪ Combine the generated data with your preferred training framework to develop robust robot learning models.<br>

> *We are creating SmilingRobo Cloud, which will allow you to train your robot using our innovative libraries and drag-and-drop facilities.*  

---

**Table of Contents**
- [Install imagination_to_real](#installing-imagination_to_real-module)
- [Image_Maker](#make-images-using-image_maker)
  - [Installation](#installation)
    - [Install ComfyUI + Dependencies](install-comfyui--dependencies)
    - [Setting up Models](setting-up-models)
  - [Usage](#usage)
    - [Running the Example Workflow](#running-the-example-workflow)
    - [Adding Your Own Workflows](#adding-your-own-workflows)
    - [Scaling Image Generation](#scaling-image-generation)
- [Create Environment](#create-environment)
  - [Installing Dependencies](#1️-installing-gym_dmc)
  - [Usage](#usage)
    - [Basic LucidSim Pipeline](#rendering-conditioning-images)
    - [Full Rendering Pipeline](#full-lucidsim-rendering-pipeline)

- [Citation](#citation)


# Installing imagination_to_real module

#### 1. Setup Conda Environment

```bash
conda create -n imagination_to_real python=3.10
conda activate imagination_to_real
git clone https://github.com/SmilingRobo/imagination-to-real imagination_to_real
cd imagination_to_real
pip install -e .

```

## Make Images using image_maker

#### 1. Install ComfyUI + Dependencies

For consistency, we recommend
using [this version](https://github.com/comfyanonymous/ComfyUI/tree/ed2fa105ae29af6621232dd8ef622ff1e3346b3f) of
ComfyUI.

```bash
# Choose the CUDA version that your GPU supports. We will use CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Installing ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
git checkout ed2fa105ae29af6621232dd8ef622ff1e3346b3f
pip install -r requirements.txt

```

#### 2. Setting up Models

We recommend placing your models outside the `ComfyUI` repo for better housekeeping. For this, you'll need to link your
model paths through a config file. Check out the `configs` folder for a template, where you'll specify locations for
checkpoints, controlnets, and VAEs. For the provided `three_mask_workflow` example, these are the models you'll need:

- [SDXL Turbo 1.0](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors): place
  under `checkpoints`
- [SDXL Depth ControlNet](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0): place under `controlnet`
- [SDXL VAE](https://huggingface.co/stabilityai/sdxl-vae): place under `vae`

After cloning this repository, you'll need to add ComfyUI to your `$PYTHONPATH` and link your model paths. We recommend
managing these in a local `.env` file. Then, link the config file you just created.

```bash
export PYTHONPATH=/path/to/ComfyUI:$PYTHONPATH

# See the `configs` folder for a template
export COMFYUI_CONFIG_PATH=/path/to/extra_model_paths.yaml
```

## Usage

imagination_to_real is organized by _workflows_. We include our main workflow called `three_mask_workflow`, which generates an image
given a depth map along with three semantic masks, each coming with a different prompt (for example,
foreground/background/object).

#### Running the Workflow

```bash
python imagination_to_real/image_maker/scripts/demo_three_mask_workflow.py [--path-to-folder] [--seed] [--save]
```

where `path-to-folder` corresponds to your data to generate images, and the `save` flag writes the output to the corresponding `examples/three_mask_workflow/[example-name]/samples` folder. The script will randomly select one of our provided prompts.

To make your data, take this example as the reference `examples/image-maker/three_mask_workflow/ramps`

##### Example

We provide example conditioning images and prompts for `three_mask_workflow` under the `examples/image-maker/three_mask_workflow` folder, grouped by scene. 

To try it out, use:

```bash
python imagination_to_real/image_maker/scripts/demo_three_mask_workflow.py [--example-name] [--seed] [--save]
```

where `example-name` corresponds to one of the scenes in the `examples/image-maker/three_mask_workflow` folder.

#### Adding Your Own Workflows

The graphical interface for ComfyUI is very helpful for designing your own workflows. Please see their documentation for
how to do this. By using this
helpful [workflow to python conversion tool](https://github.com/pydn/ComfyUI-to-Python-Extension.git), you can script
your workflows as we've done with `Image_Maker/workflows/three_mask_workflow.py`.

#### Scaling Image Generation

In LucidSim, we use a distributed setup to generate images at scale. We utilize rendering nodes, launched independently
on many machines, that receive and fulfill rendering requests from the physics engine containing prompts and
conditioning images through a task queue (see [Zaku](https://zaku.readthedocs.io/en/latest/)). We hope to release setup
instructions for this in the future, but we have included `Image_Maker/render_node.py` for your reference.

---

## Create Environment

<table style="border-collapse: collapse; border: none; width: 100%;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="assets/images/example_conditioning.png" style="width: 500px; max-width: 100%;" /><br>
    </td>
    <td style="text-align: center; border: none;">
      <img src="assets/images/example_imagen.png" style="width: 500px; max-width: 100%;" /><br>
    </td>
  </tr>
</table>

#### 1.Installing gym_dmc

The last few dependencies require a downgraded `setuptools` and `wheel` to install. To install, please downgrade and
revert after.

```bash
pip install setuptools==65.5.0 wheel==0.38.4 pip==23
pip install gym==0.21.0
pip install gym-dmc==0.2.9
pip install -U setuptools wheel pip
```

#### Usage

**Note:** On Linux, make sure to set the environment variable ` MUJOCO_GL=egl`.

LucidSim generates photorealistic images by using a generative model to augment the simulator's rendering, using
conditioning images to maintain control over the scene geometry.

#### Rendering Conditioning Images

We have provided an expert policy checkpoint under `checkpoints/expert.pt`. This policy was derived from that
of [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour). You can use this policy to sample an environment
and visualize the conditioning images with:

#### If you are using custom data

> if you are follwing example just run the script

1. Create a `name.py` file in `imagination_to_real/specs` take `gaps.py` as reference, change line `5`.

2. create your `name`.py and `name`.xml file in `imagination_to_real/lucidsim/tasks`. Take the `gaps.py` and `gaps.xml` as reference and just change the line `11` of .py and `3` of .xml
> if you are using your own robot then you have to change line `2` of xml too.

3. Make you `name`.xml file and add it here `imagination_to_real/lucidsim/tasks/assets/terrains`. take the `gaps.xml` as reference


```bash
# example env-name: one of ['parkour', 'hurdle', 'gaps', 'stairs_v1', 'stairs_v2']
!python imagination_to_real/lucidsim/scripts/play.py --save-path [--env-name] [--num-steps] [--seed]
````

where `save_path` is where to save the resulting video.

#### Full Rendering Pipeline

To run the full generative augmentation pipeline, please also make sure the environment variables are still
set correctly:

```bash
COMFYUI_CONFIG_PATH=/path/to/extra_model_paths.yaml
PYTHONPATH=/path/to/ComfyUI:$PYTHONPATH
```


You can then run the full pipeline with:

```bash
python imagination_to_real/lucidsim/scripts/play_three_mask_workflow.py --save-path --prompt-collection [--env-name] [--num-steps] [--seed]
```

where `save_path` and `env_name` are the same as before. `prompt_collection` should be a path to a `.jsonl` file with
correctly formatted prompts, as in the `imagination-to-real/tree/master/examples/image-maker/three_mask_workflow` folder.

---

We thank the authors of [LucidSim](https://github.com/lucidsim/lucidsim) for their opensource code and [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour) for their open-source codebase, which we used as a starting point for our library.



## Citation

If you find our work useful, please consider citing:

```
@inproceedings{yu2024learning,
  title={Learning Visual Parkour from Generated Images},
  author={Alan Yu and Ge Yang and Ran Choi and Yajvan Ravan and John Leonard and Phillip Isola},
  booktitle={8th Annual Conference on Robot Learning},
  year={2024},
}
```
