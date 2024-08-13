import os
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download

def prepare_base_model():
    print(f'Preparing base stable-diffusion-v1-5 weights...')
    local_dir = "./checkpoints/stable-diffusion-v1-5"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["unet/config.json", "unet/diffusion_pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )
        os.system(f"rm -rf {local_dir}/.cache")

def prepare_image_encoder():
    print(f"Preparing image encoder weights...")
    local_dir = "./checkpoints/image_encoder"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        #breakpoint()
        hf_hub_download(
            repo_id="jeeyoung/sim2real_image_enc_unet",
            filename=path.name,
            local_dir=local_dir,
        )
        os.system(f"rm -rf {local_dir}/.cache")

def prepare_denoising_unet():
    print(f"Preparing denoising unet weights...")
    local_dir = "./checkpoints/denoising_unet"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "config.json",
        "diffusion_pytorch_model.safetensors",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="jeeyoung/sim2real_denoising_unet",
            filename=path.name,
            local_dir=local_dir,
        )
        os.system(f"rm -rf {local_dir}/.cache")

def prepare_pose_guider():
    print(f"Preparing pose_guider ...")
    local_dir = "./checkpoints/pose_guider"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "pose_guider.pth"
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="jeeyoung/sim2real_pose_guider",
            filename=path.name,
            local_dir=local_dir,
        )
        os.system(f"rm -rf {local_dir}/.cache")
    

def prepare_reference_unet():
    print(f"Preparing reference unet weights...")
    local_dir = "./checkpoints/reference_unet"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "config.json",
        "diffusion_pytorch_model.safetensors",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="jeeyoung/sim2real_ref_unet",
            filename=path.name,
            local_dir=local_dir,
        )
        os.system(f"rm -rf {local_dir}/.cache")

if __name__ == '__main__':
    #prepare_base_model()
    #prepare_denoising_unet()
    #prepare_reference_unet()
    #prepare_image_encoder()
    prepare_pose_guider()