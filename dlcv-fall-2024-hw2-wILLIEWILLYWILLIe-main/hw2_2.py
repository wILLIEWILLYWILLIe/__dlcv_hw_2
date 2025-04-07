import os
import yaml
import argparse
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import torch.multiprocessing as tormp
tormp.set_start_method('spawn', force=True)  

from UNet import UNet

P2_diffusion_params = {
  "num_timesteps" : 1000,
  "beta_start"  : 0.00001,
  "beta_end"    : 0.02,
  "total_steps" : 50,
  "batch_size"  : 10,
  "eta"     : 0,
  "pred"    : 0,
  "use_gpu" : 1
}

def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    return torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float32)

class FaceDataset(Dataset):
    def __init__(self, noise_dir):
        self.noise_dir = noise_dir
        self.noise_files = sorted([f for f in os.listdir(noise_dir) if f.endswith('.pt')])
        print(f"Noise in noise dir: {self.noise_files}")
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.noise_files)

    def __getitem__(self, idx):
        noise = torch.load(os.path.join(self.noise_dir, self.noise_files[idx]))
        return noise

class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda", total_steps=None, batch_size = 1, eta = 0):
        
        self.num_timesteps  = num_timesteps
        self.device         = device
        self.batch_size     = batch_size

        # Setup Diffusion coefficient
        self.betas          = beta_scheduler(num_timesteps, beta_start, beta_end).to(device)
        self.alphas         = 1. - self.betas
        self.alpha_hat      = torch.cumprod(self.alphas, dim=0)

        # Setup the DDIM timesteps
        self.total_steps    = total_steps if total_steps is not None else num_timesteps
        self.ddim_timestep  = self.__generate_timesteps()

        # Compute DDIM coefficients
        self.sigma, self.coeff, self.sqrt_alpha_i_min_1 = self.__ddim_coeff_compute(eta)

        # Reverse for proper sampling order
        self.sigma = list(self.sigma.flip(0))
        self.coeff = list(self.coeff.flip(0))
        self.sqrt_alpha_i_min_1 = list(self.sqrt_alpha_i_min_1.flip(0))

    def __generate_timesteps(self):
        return np.linspace(0, self.num_timesteps - 1, self.total_steps, dtype=int)

    def __ddim_coeff_compute(self, eta):
        """
        Compute DDIM sampling coefficients based on alpha values.
        :param eta: Controls the stochasticity, eta = 0 for deterministic DDIM sampling.
        :return: sigma, coeff, sqrt_alpha_i_min_1
        """
        # Alpha values at timesteps tau
        alpha_tau_i         = self.alpha_hat[self.ddim_timestep]
        alpha_tau_i_min_1   = F.pad(self.alpha_hat[self.ddim_timestep[:-1]], pad=(1, 0), value=1.0)

        # Sigma for stochasticity control
        sigma               = eta * torch.sqrt((1 - alpha_tau_i_min_1) / (1 - alpha_tau_i) * (1 - alpha_tau_i / alpha_tau_i_min_1))
        coeff               = torch.sqrt(1 - alpha_tau_i_min_1 - sigma ** 2)
        sqrt_alpha_i_min_1  = torch.sqrt(alpha_tau_i_min_1)

        return sigma, coeff, sqrt_alpha_i_min_1


    def ddim_sample(self, model, noise):
        save_timesteps  = [999, 897, 795, 693, 591, 489, 387, 285, 183, 81, 0]
        save_timesteps  = [-1]
        all_timesteps   = []
        
        model.eval()
        x = noise.to(self.device)
        if x.dim() == 5: 
            x = x.squeeze(1)
        curr_batch = x.size(0)
        # print(f"ddim_sample shape: {x.shape}, Dtype: {x.dtype}, Device: {x.device}")
        
        reversed_timesteps = list(reversed(self.ddim_timestep))
        with torch.no_grad():
            for i, t_num in enumerate(reversed_timesteps):
                t = (torch.ones(curr_batch) * t_num).long().to(self.device)  
                predicted_noise = model(x, t)
            
                alpha       = self.alphas[t][:, None, None, None]
                alpha_hat   = self.alpha_hat[t][:, None, None, None]

                if i < len(self.ddim_timestep) - 1:
                    next_t = reversed_timesteps[i + 1]
                else:
                    next_t = t_num 
                print(f'current t {t_num} next t {next_t}', end = '\r')

                next_t          = (torch.ones(curr_batch) * next_t).long().to(self.device)  
                next_alpha      = self.alphas[next_t][:, None, None, None]
                next_alpha_hat  = self.alpha_hat[next_t][:, None, None, None]
                x0_pred         = (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)

                if t_num > 0:
                    # Apply stochasticity based on eta
                    noise = torch.randn_like(x) #if eta > 0 else torch.zeros_like(x)
                    x = self.sqrt_alpha_i_min_1[i] * x0_pred + self.coeff[i] * predicted_noise + self.sigma[i] * noise
                else:
                    x = x0_pred

                if t_num in save_timesteps : 
                    all_timesteps.append(((x.clone().clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8))

        x = (x.clamp(-1, 1) + 1) / 2
        return x

def mse_from_path(generated_image_path, gt_image_path, device):
    mse_criterion = nn.MSELoss()

    generated_image_pil = Image.open(generated_image_path)
    gt_image_pil        = Image.open(gt_image_path)

    transform_to_tensor     = transforms.ToTensor()
    generated_image_tensor  = transform_to_tensor(generated_image_pil).to(device)
    gt_image_tensor         = transform_to_tensor(gt_image_pil).to(device)

    assert generated_image_tensor.shape == gt_image_tensor.shape, \
        f"Shape mismatch: {generated_image_tensor.shape} vs {gt_image_tensor.shape}"

    # Scale to [0, 255]
    generated_image_tensor = (generated_image_tensor * 255)
    gt_image_tensor = (gt_image_tensor * 255)
    mse = mse_criterion(generated_image_tensor, gt_image_tensor)

    return mse.item()

def ddim_generate_images(input_noise_dir, output_images_dir, model_weight):

    if not os.path.isdir(input_noise_dir):
        raise FileNotFoundError(f"The input noise directory '{input_noise_dir}' does not exist.")
    if not os.path.exists(model_weight):
        raise FileNotFoundError(f"The model weight file '{model_weight}' does not exist.")
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    device = "cuda" if torch.cuda.is_available() and P2_diffusion_params["use_gpu"] else "cpu"

    # Load the model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_weight, map_location=device))

    diffusion_main      = Diffusion(
        num_timesteps   =P2_diffusion_params["num_timesteps"], 
        beta_start      =P2_diffusion_params["beta_start"], 
        beta_end        =P2_diffusion_params["beta_end"], 
        device          =device,
        total_steps     =P2_diffusion_params["total_steps"],
        batch_size      =P2_diffusion_params["batch_size"],
        eta             =P2_diffusion_params["eta"]
    )

    face_dataset    = FaceDataset(input_noise_dir)
    face_loader     = DataLoader(face_dataset, batch_size = P2_diffusion_params["batch_size"], shuffle = False, num_workers = 4)

    start_sample_time = time.time()
    for batch_idx, noise_input_batch in enumerate(face_loader):
        noise_input_batch           = noise_input_batch.to(device)
        generated_images_batch      = diffusion_main.ddim_sample(model, noise_input_batch)

        for i in range(noise_input_batch.size(0)):
            save_path       = os.path.join(output_images_dir, f"{batch_idx * noise_input_batch.size(0) +  i:02}.png")
            generated_image = generated_images_batch[i]
            generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())
            save_image(generated_image, save_path, normalize=False) 

            print(f"Saved image->{save_path}")
    print(f'Finish DDIM Sampling in {time.time() - start_sample_time:.3f} sec')
    torch.cuda.empty_cache()

    pred        = P2_diffusion_params['pred']
    GT_folder   = "./hw2_data/face/GT"
    if pred and os.path.exists(GT_folder):
        
        total_mse       = 0
        gt_files        = sorted([f for f in os.listdir(GT_folder) if f.endswith('.png')])
        generated_files = sorted([f for f in os.listdir(output_images_dir) if f.endswith('.png')])

        for gt_file, gen_file in zip(gt_files, generated_files):
            gt_path     = os.path.join(GT_folder, gt_file)
            gen_path    = os.path.join(output_images_dir, gen_file)
            mse         = mse_from_path(gen_path, gt_path, device)
            total_mse   += mse
            print(f"MSE for {gen_file}: {mse}")

        avg_mse = total_mse / len(gt_files)
        print(f"Total MSE: {total_mse:.4f}, Average MSE: {avg_mse:.4f}")
        return avg_mse
    else:
        # print("No ground truth folder provided. Skipping MSE calculation.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDIM Inplementation.")
    parser.add_argument('--input_noise_dir', type=str, required=True, help="Directory containing the predefined noise files")
    parser.add_argument('--output_images_dir', type=str, required=True, help="Directory where the generated images will be saved")
    parser.add_argument('--model_weight', type=str, required=True, help="Path to the pretrained model weights")

    args = parser.parse_args()

    ddim_generate_images(args.input_noise_dir, args.output_images_dir, args.model_weight)
    