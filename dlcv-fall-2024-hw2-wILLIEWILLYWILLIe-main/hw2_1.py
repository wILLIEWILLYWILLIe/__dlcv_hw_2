import os
import random
import time
import argparse
import torch
from MY_UTILS.P1_train import Diffusion, sample_images_mode
from MY_UTILS.P1_Unet import UNet

config = {
    "P1_dataset_params": {
        "MNISTM_OUTPUT_DIR" : "mnistm",
        "SVHN_OUTPUT_DIR"   : "svhn"
    },
    "P1_diffusion_params": {
        "num_timesteps" : 1000,
        "beta_start"    : 0.0001,
        "beta_end"      : 0.02
    },
    "P1_model_params": {
        "im_size": 32
    },
    "P1_train_params": {
        "task_name"     : 'P1_DDPM',
        "num_classes"   : 20,
        "num_samples"   : 50,
        "load_ckpt_name": "./hw2_ckpt/P1_ddpm2.pth",
        "load_ckpt"     : True,  # Load the above checkpoint for pretrain or predict
        "sample_only"   : True,  # Call the sampling function to generate images only
        "mode"          : "inf",
        "sample_once"   : 1,
    }
}

def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def main(outputDir_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_config  = config["P1_dataset_params"]
    model_config    = config["P1_model_params"]
    train_config    = config["P1_train_params"]
    diffusion_config= config["P1_diffusion_params"]

    MNISTM_OUTPUT_DIR   = os.path.join(outputDir_path, dataset_config["MNISTM_OUTPUT_DIR"])
    SVHN_OUTPUT_DIR     = os.path.join(outputDir_path, dataset_config["SVHN_OUTPUT_DIR"])

    check_and_create_directory(MNISTM_OUTPUT_DIR)
    check_and_create_directory(SVHN_OUTPUT_DIR)

    diffusion   = Diffusion(
        num_timesteps   = diffusion_config["num_timesteps"], 
        beta_start      = diffusion_config["beta_start"], 
        beta_end        = diffusion_config["beta_end"], 
        imgz            = model_config["im_size"]
    )

    model       = UNet(num_classes = train_config["num_classes"]).to(device)

    if train_config["load_ckpt"] and os.path.exists(train_config["load_ckpt_name"]):
        print(f"Loading checkpoint from {train_config['load_ckpt_name']}\n", end = '\r')
        try:
            checkpoint = torch.load(train_config["load_ckpt_name"], map_location=device)
            model.load_state_dict(checkpoint)
            print("--> Checkpoint loaded successfully.")
        except Exception as e:
            print(f"--> Error loading checkpoint: {e}")
            
    model.eval()
    sampling_time = time.time()
    for dataset_name, save_path in [
                                    ['mnistm', MNISTM_OUTPUT_DIR], 
                                    ['svhn', SVHN_OUTPUT_DIR]
                                    ]:
        sample_images_mode( diffusion, 
                            model, 
                            device, 
                            sample_num  = train_config["num_samples"], 
                            mode        = train_config["mode"],
                            dataset_name= dataset_name,
                            save_path   = save_path,
                            sample_once = train_config["sample_once"]
        )
    print(f'Using {time.time() - sampling_time:.3f} to sample images')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion Image Generator')
    parser.add_argument('--outputDir', type=str, default='./MY_UTILS/config.yaml', help='Path to the output directory')
    args = parser.parse_args()
    main(outputDir_path=args.outputDir)
