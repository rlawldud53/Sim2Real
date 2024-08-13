## Environment
```
git clone https://github.com/rlawldud53/Sim2Real.git
cd Sim2Real

conda env create -f environment.yaml
conda activate sim2real
pip install -r requirements.txt
```
## Checkpoints
**Automatically downloading**: You can run the following command to download pretrained checkpoints automatically: 
```shell
python tools/download_weights.py
```
Checkpoints will be placed under the `./checkpoints` directory. The whole downloading process may take a long time.

**Manually downloading**: You can also download pretrained checkpoints manually [here](https://drive.google.com/drive/folders/1LEN9Eq1TQ7bi--NjEQyK1iky9tgQBvvN?usp=sharing):

checkpoints should be orgnized as follows:
```
checkpoints/
├── denoising_unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── image_encoder/
│   ├── config.json
│   └── pytorch_model.bin
├── pose_guider/
│   └── pose_guider.pth
├── reference_unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── sd-image-variations-diffusers/unet/
    └── config.json
```

## Inference
### Data Preparation
We provide example images which you can find in the test_imgs directory. The data structure is outlined below:
```
Sim2Real/
├── test_imgs/
│   └── condition/
│       ├── canny/
│       │   ├── frankfurt_000000_003025.png
│       │   ├── landau_000014_000019.png
│       │   ├── munster_000033_000019.png
│       │   ├── munster_000038_000019.png
│       │   └── munster_000127_000019.png
│       └── panoptic/
│           ├── frankfurt_000000_001751.png
│           ├── frankfurt_000000_003025.png
│           ├── landau_000014_000019.png
│           ├── munster_000033_000019.png
│           ├── munster_000038_000019.png
│           └── munster_000127_000019.png
├── reference/
│   ├── frankfurt_000000_001016.json
│   ├── frankfurt_000000_001016.png
│   ├── leverkusen_000013_000019.json
│   ├── leverkusen_000013_000019.png
│   ├── lindau_000008_000019.json
│   └── lindau_000008_000019.png
└── seg_map/
    ├── frankfurt_000000_001751.json
    ├── frankfurt_000000_001751.png
    ├── frankfurt_000000_003025.json
    ├── frankfurt_000000_003025.png
    ├── landau_000014_000019.json
    ├── landau_000014_000019.png
    ├── munster_000033_000019.json
    ├── munster_000033_000019.png
    ├── munster_000038_000019.json
    ├── munster_000038_000019.png
    ├── munster_000127_000019.json
    └── munster_000127_000019.png
```

If you wish to use your own data, place your chosen style images and corresponding JSON files in the `test_imgs/reference` folder.

In the `test_imgs/seg_map` folder, place the structure images you want to transfer that style to, along with their related JSON files formatted as segmentation maps. 
The JSON files should contain information about the objects within the images. For detailed specifics, please refer to the format of the example JSON files we have provided

Additionally, in the `test_imgs/condition` folder, place images that have been converted from those segmentation maps to either panoptic or canny forms. 
These images should use the same filenames as those used in the `test_imgs/seg_map` folder. 

You are free to choose between using canny or panoptic styles for the condition images, based on the condition you wish to apply.

### Inference 
Here we provide inference scripts. Just type following command to run inference
```bash
bash scripts/inference.sh
```
Or you can inference the images by directly specifying the path or conditions in the command below
```bash
python inference.py
--config ./configs/inference/inference_pose2img.yaml \
--ref_path ./test_imgs/reference/YOUR_REFERENCE_IMG_FILENAME.png \
--seg_path ./test_imgs/seg_map/YOUR_SEGMENTATION_IMG_FILENAME.png \
--output_dir ./outputs \
--cond_mode multi_panop \
--width 256 \
--height 256 \
--seed 42 \ 
```
You can choose either `multi_panop` or `multi_canny` for the cond_mode.
