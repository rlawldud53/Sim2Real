import os
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download

def prepare_weights():
    print(f"Preparing weights...")

    local_dir = "./checkpoints/"
    dirname = ["denoising_unet","reference_unet","pose_guider","","","",""]
    download_filename = [
        "denoising_unet.pth",
        "reference_unet.pth",
        "pose_guider.pth",
        "sd-image-variations-diffusers/unet/config.json",
        "sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin",
        "image_encoder/config.json",
        "image_encoder/pytorch_model.bin"
    ]
    for d, hub_file in zip(dirname,download_filename):
        #breakpoint()
        saved_path = os.path.join(local_dir,d)
        
        if os.path.exists(saved_path):
            continue
        else :
            os.makedirs(saved_path, exist_ok=True)
        hf_hub_download(
            repo_id="jeeyoung/sim2real_weights",
            filename=hub_file,
            local_dir=local_dir,
        )
        os.system(f"rm -rf {saved_path}/.cache")

if __name__ == '__main__':
    prepare_weights()
