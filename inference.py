import argparse
import os
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from src.models.unet_2d_condition import UNet2DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from src.models.pose_guider import PoseGuider
from tqdm import tqdm
import glob

def main(args):
    cfg = OmegaConf.load(args.config)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else: ## yaml -> fp32로 설정 안하면 에러
        weight_dtype = torch.float32

    # initiliaze network, ecncoder, guider
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype,device="cuda")

    denoising_unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype,device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(
            conditioning_embedding_channels=320, 
            condition_num=1 if args.cond_mode == "single" else 2
        ).to(device="cuda")
    
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    inference_config_path = cfg.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)
    
    generator = torch.manual_seed(args.seed)

    width, height = args.width, args.height
    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(cfg.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(cfg.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(cfg.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    ).to("cuda",dtype=weight_dtype)

    pil_images = []

    if args.ref_path == "":
        ref_lst = [x for x in cfg["test_cases"].keys() if x != 'default']
    else:
        ref_lst = [args.ref_path]
    
    for ref_image_path in tqdm(ref_lst):
        print("Running inference...")
        print(f"############ Current Reference image : {ref_image_path.split('/')[-1]} ############")
        
        if args.ref_path == "":
            pose_lst = cfg["test_cases"][ref_image_path]
        else:
            pose_lst = [args.seg_path]
        
        assert len(pose_lst) != 0, "No segmentation map image exists!"
        
        for pose_image_path in pose_lst:
            ref_name = ref_image_path.split("/")[-1].replace(".png", "")
            pose_name = pose_image_path.split("/")[-1].replace(".png", "")

            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            pose_image_pil = Image.open(pose_image_path).convert("RGB")

            cond_image_pil = None
            pose_img_name = pose_image_path.split("/")[-1].split(".")[0]
            cond_image_name = []
            if args.cond_mode == "multi_canny":
                cond_image_name = glob.glob(f"./test_imgs/condition/canny/{pose_img_name}.*")
            elif args.cond_mode == "multi_panop":
                cond_image_name = glob.glob(f"./test_imgs/condition/panoptic/{pose_img_name}.*")
            
            assert len(cond_image_name) != 0, "No matching condition image exists!"
            cond_image_pil = Image.open(cond_image_name[0]).convert("RGB")
            
            pose_json = pose_image_path.replace(".png", ".json")
            ref_json = ref_image_path.replace(".png", ".json")

            
            image = pipe(
                ref_image_pil,
                pose_image_pil,
                cond_image_pil,
                ref_json,
                pose_json,
                args.width,
                args.height,
                20,
                1.0,
                generator=generator,
            ).images

            image = image[0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
            res_image_pil = Image.fromarray((image * 255).astype(np.uint8))

            w, h = res_image_pil.size
            if cond_image_pil is not None:
                canvas = Image.new("RGB", (w * 4, h), "white")
                ref_image_pil = ref_image_pil.resize((w, h))
                pose_image_pil = pose_image_pil.resize((w, h))
                cond_image_pil = cond_image_pil.resize((w ,h ))
                canvas.paste(ref_image_pil, (0, 0))
                canvas.paste(pose_image_pil, (w, 0))
                canvas.paste(cond_image_pil, (w * 2, 0))
                canvas.paste(res_image_pil, (w * 3, 0))
            else:
                canvas = Image.new("RGB", (w * 3, h), "white")
                ref_image_pil = ref_image_pil.resize((w, h))
                pose_image_pil = pose_image_pil.resize((w, h))
                canvas.paste(ref_image_pil, (0, 0))
                canvas.paste(pose_image_pil, (w, 0))
                canvas.paste(res_image_pil, (w * 2, 0))


            pil_images.append({"ref_name": ref_name, "pose_name": pose_name,"img": canvas})
            
    for img in tqdm(pil_images):
        save_ref_dir = f"{args.output_dir}/{img['ref_name']}"
        if not os.path.exists(save_ref_dir) : os.makedirs(save_ref_dir)
        img['img'].save(f"{save_ref_dir}/{img['pose_name']}_{args.cond_mode}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    # config.data -> 256,256
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=784)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,default="./results")

    parser.add_argument("--ref_path",type=str,default="./test_imgs/reference/frankfurt_000000_001016.png")
    parser.add_argument("--seg_path",type=str,default="./test_imgs/seg_map/lindau_000014_000019.png")

    parser.add_argument("--cond_mode",type=str,default="multi_canny")
    args, unknown = parser.parse_known_args()

    main(args)



