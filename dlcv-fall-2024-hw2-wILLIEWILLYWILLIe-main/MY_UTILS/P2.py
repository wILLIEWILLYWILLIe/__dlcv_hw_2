import os
import yaml
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader

from P2_UNet import UNet

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float32)
    return betas

def beta_scheduler_cosine(n_timestep=1000):
    t = torch.linspace(0, n_timestep, n_timestep)
    betas = (1 - torch.cos((t / n_timestep) * (3.14159265359 / 2)))**2
    return betas

def save_images_grid_timestep(all_timesteps, output_path="generated_images_grid.png"):
    """
    Save images as a grid using PIL where rows are different images in the batch and columns are different timesteps.
    
    :param all_timesteps: List of images for each timestep. Each element is a tensor of shape (batch_size, C, H, W).
    :param output_path: Path to save the output image grid.
    """
    to_pil = transforms.ToPILImage()

    all_timesteps_np = [img.cpu() for img in all_timesteps]
    batch_size, C, H, W = all_timesteps_np[0].shape
    num_timesteps = len(all_timesteps_np)

    # Create a blank canvas for the grid
    grid_img = Image.new('RGB', (num_timesteps * W, batch_size * H))

    for row in range(batch_size):
        for col in range(num_timesteps):
            img = all_timesteps_np[col][row]  # Shape (C, H, W)
            img_pil = to_pil(img)  # Convert to PIL image

            grid_img.paste(img_pil, (col * W, row * H))

    grid_img.save(output_path)

def save_images_grid_eta(eta_images_list, etas, save_path):
    """
    Save the images for varying eta values in a grid format using PIL.
    :param eta_images_list: List of generated images for each eta
    :param etas: List of eta values corresponding to the images
    :param save_path: Path to save the final grid image
    """
    batch_size, C, H, W = eta_images_list[0].shape
    to_pil = transforms.ToPILImage()

    # Create a list to hold each row (for each eta)
    rows = []

    # For each eta, concatenate the batch images into a row
    for i, images in enumerate(eta_images_list):
        row_images = []
        for j in range(batch_size):
            img_tensor = images[j]
            img_pil = to_pil(img_tensor)  # Convert tensor to PIL image
            row_images.append(img_pil)
        
        # Concatenate the images horizontally to form a row
        row = concatenate_images(row_images, direction='horizontal')
        rows.append(row)

    # Concatenate all rows vertically
    grid_image = concatenate_images(rows, direction='vertical')

    # Save the final grid image
    grid_image.save(save_path)
    print(f"Saved image grid with varying eta to {save_path}")

def concatenate_images(images, direction='horizontal'):
    """
    Concatenate a list of PIL images in the specified direction (horizontal or vertical).
    :param images: List of PIL images
    :param direction: Direction to concatenate ('horizontal' or 'vertical')
    :return: Concatenated PIL image
    """
    if direction == 'horizontal':
        # Concatenate horizontally
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_img = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width

    elif direction == 'vertical':
        # Concatenate vertically
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)
        new_img = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for img in images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.height

    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return new_img

class FaceDataset(Dataset):
    def __init__(self, noise_dir, gt_dir):
        self.noise_dir  = noise_dir
        self.gt_dir     = gt_dir

        self.noise_files    = sorted([f for f in os.listdir(noise_dir) if f.endswith('.pt')])
        self.gt_files       = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
        self.transform      = transforms.ToTensor() 
        # print(f'Noise Files: {self.noise_files}\nGT Files: {self.gt_files}')

    def __len__(self):
        return len(self.noise_files)

    def __getitem__(self, idx):

        noise_path  = os.path.join(self.noise_dir, self.noise_files[idx])
        noise_pt    = torch.load(noise_path)
        gt_path     = os.path.join(self.gt_dir, self.gt_files[idx])
        gt_image    = Image.open(gt_path).convert('RGB')
        gt_image    = self.transform(gt_image)

        return noise_pt, gt_image
    
class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda", total_steps=None, batch_size = 1, eta = 0):
        
        self.num_timesteps  = num_timesteps
        self.device         = device
        self.batch_size     = batch_size

        # Setup Diffusion coefficient
        self.betas          = beta_scheduler(num_timesteps, beta_start, beta_end).to(device)
        # self.betas          = beta_scheduler_cosine(num_timesteps).to(device)
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
        sigma = eta * torch.sqrt((1 - alpha_tau_i_min_1) / (1 - alpha_tau_i) * (1 - alpha_tau_i / alpha_tau_i_min_1))
        coeff = torch.sqrt(1 - alpha_tau_i_min_1 - sigma ** 2)
        sqrt_alpha_i_min_1 = torch.sqrt(alpha_tau_i_min_1)

        return sigma, coeff, sqrt_alpha_i_min_1


    def ddim_sample(self, model, noise):
        save_timesteps = [999, 897, 795, 693, 591, 489, 387, 285, 183, 81, 0]
        all_timesteps = []
        
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
        x = (x * 255).type(torch.uint8)
        return x, all_timesteps

    def ddpm_sample(self, model, noise, deterministic=False, seed=100):
        save_timesteps  = [999, 897, 795, 693, 591, 489, 387, 285, 183, 81, 0]
        all_timesteps   = []

        model.eval()
        x = noise.to(self.device)
        if x.dim() == 5: 
            x = x.squeeze(1)
        curr_batch = x.size(0)
        # print(f"ddpm_sample shape: {x.shape}, Dtype: {x.dtype}, Device: {x.device}")

        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            for i in reversed(range(0, self.num_timesteps)):
                t = (torch.ones(curr_batch) * i).long().to(self.device) # The current timestep for all `n` images
                # --> Runs over all timesteps(from last to first), less noisy than previous
                # print(f"Shape: {x.shape}, Dtype: {x.dtype}, t_Dtype: {t.dtype}")
                predicted_noise = model(x, t)

                alpha       = self.alphas[t][:,None, None, None]
                alpha_hat   = self.alpha_hat[t][:,None, None, None]
                beta        = self.betas[t][:,None, None, None]

                if not deterministic and i > 1:
                    noise = torch.randn_like(x) 
                else:
                    noise = torch.zeros_like(x) # No noise for the last step
                x = 1/torch.sqrt(alpha) * (x-((1-alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if i in save_timesteps : 
                    all_timesteps.append(((x.clone().clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8))

        model.train()
        x = (x.clamp(-1,1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x, all_timesteps

class Interpolator:
    def __init__(self, model, diffusion, device="cuda"):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.to_pil = transforms.ToPILImage()

    def slerp(self, val, low, high):

        # Flatten the tensors to apply cosine similarity
        low_flat = low.view(-1)  
        high_flat = high.view(-1) 
        
        dot_product = F.cosine_similarity(low_flat.unsqueeze(0), high_flat.unsqueeze(0), dim=-1)
        omega = torch.acos(dot_product.clamp(-1, 1))  # Clamp to handle any floating-point errors

        so = torch.sin(omega)
        if torch.all(so == 0):
            # If the angle is very small, fall back to linear interpolation
            return (1.0 - val) * low + val * high

        # Perform SLERP on the flattened tensors
        interpolated_flat = (torch.sin((1.0 - val) * omega) / so) * low_flat + (torch.sin(val * omega) / so) * high_flat
        
        # Reshape the interpolated result back to the original shape of `low` and `high`
        return interpolated_flat.view_as(low)

    def linear_interpolation(self, val, low, high):
 
        return (1.0 - val) * low + val * high

    def generate_interpolated_noises(self, noise_0, noise_1, interpolation_func, steps=11):
        interpolated_noises = []
        alphas = np.linspace(0.0, 1.0, steps)

        for alpha in alphas:
            noise_interpolated = interpolation_func(alpha, noise_0, noise_1)
            interpolated_noises.append(noise_interpolated.to(self.device))

        return torch.stack(interpolated_noises)

    def generate_images_from_noises(self, noise_vectors):

        batch_size = noise_vectors.size(0)  # Expecting [steps, 3, 256, 256]
        generated_images_batch, _ = self.diffusion.ddim_sample(self.model, noise_vectors)

        generated_images = [generated_images_batch[i].cpu() for i in range(batch_size)]

        return generated_images

    def save_interpolated_images_grid(self, interpolated_images, save_path):

        images_pil = [self.to_pil(img.squeeze(0)) for img in interpolated_images]
        grid_image = self.concatenate_images(images_pil, direction='horizontal')
        grid_image.save(save_path)

    def concatenate_images(self, images, direction='horizontal'):

        if direction == 'horizontal':
            # Concatenate horizontally
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            new_img = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for img in images:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.width

        elif direction == 'vertical':
            # Concatenate vertically
            widths, heights = zip(*(i.size for i in images))
            max_width = max(widths)
            total_height = sum(heights)
            new_img = Image.new('RGB', (max_width, total_height))

            y_offset = 0
            for img in images:
                new_img.paste(img, (0, y_offset))
                y_offset += img.height

        else:
            raise ValueError(f"Unsupported direction: {direction}")

        return new_img

    def generate_and_save(self, noise_0, noise_1, steps=11, save_paths=('slerp.png', 'linear.png')):

        # Generate SLERP interpolated noises first
        print("Generating SLERP interpolation...")
        slerp_noises = self.generate_interpolated_noises(noise_0, noise_1, self.slerp, steps=steps)
        slerp_images = self.generate_images_from_noises(slerp_noises)
        self.save_interpolated_images_grid(slerp_images, save_paths[0])

        # Generate Linear interpolated noises first
        print("Generating Linear interpolation...")
        linear_noises = self.generate_interpolated_noises(noise_0, noise_1, self.linear_interpolation, steps=steps)
        linear_images = self.generate_images_from_noises(linear_noises)
        self.save_interpolated_images_grid(linear_images, save_paths[1])

def mse_from_path(generated_image_path, gt_image_path, device):
    mse_criterion = nn.MSELoss()

    generated_image_pil = Image.open(generated_image_path)
    gt_image_pil        = Image.open(gt_image_path)

    transform_to_tensor     = transforms.ToTensor()
    generated_image_tensor  = transform_to_tensor(generated_image_pil).to(device)
    gt_image_tensor         = transform_to_tensor(gt_image_pil).to(device)

    assert generated_image_tensor.shape == gt_image_tensor.shape, \
        f"Shape mismatch: {generated_image_tensor.shape} vs {gt_image_tensor.shape}"

    mse = mse_criterion(generated_image_tensor, gt_image_tensor)

    return mse.item()

def evaluate_ddim_model(diffusion, model, face_loader, save_dir, device="cuda", timestep_save_path = None):

    mse_criterion       = nn.MSELoss()
    total_mse           = 0
    count               = 0

    for batch_idx, (noise_input_batch, image_gt_batch) in enumerate(face_loader):

        noise_input_batch = noise_input_batch.to(device)
        print(f"Batch shape: {noise_input_batch.shape}, Dtype: {noise_input_batch.dtype}, Device: {noise_input_batch.device}")

        generated_images_batch, all_images = diffusion.ddim_sample(model, noise_input_batch)
        # generated_images_batch, all_images = diffusion.ddpm_sample(model, noise_input_batch)
        print(f"Generated batch shape: {generated_images_batch.shape}, Dtype: {generated_images_batch.dtype}")
        if timestep_save_path is not None:
            save_images_grid_timestep(all_images, output_path=timestep_save_path)

        for i in range(noise_input_batch.size(0)):
            save_path       = os.path.join(save_dir, f"{batch_idx *noise_input_batch.size(0) +  i:02}.png")
            gt_save_path    = os.path.join(save_dir, f"{batch_idx *noise_input_batch.size(0) +  i:02}_gt.png")

            generated_image = generated_images_batch[i]
            gt_image = image_gt_batch[i]

            # Save generated image
            pil_image = transforms.ToPILImage()(generated_image.cpu())
            pil_image.save(save_path)

            # Save ground truth image
            gt_image_pil = transforms.ToPILImage()(gt_image.cpu())
            gt_image_pil.save(gt_save_path)

            # Calculate MSE for each image
            # mse = mse_criterion(generated_image, gt_image.to(device))
            mse         = mse_from_path(save_path, gt_save_path, device)
            total_mse   += mse
        
        count += noise_input_batch.size(0)
        del noise_input_batch
        del image_gt_batch
        torch.cuda.empty_cache()

    if count > 0:
        avg_mse = total_mse / count
        print(f"Total MSE: {total_mse:.3f}")
        return avg_mse
    else:
        print("No valid images processed.")
    return None

def vary_eta_generate(model, face_loader_eta, diffusion_config = None, device = "cuda", etas = [0.0], save_path = ''):
    eta_images_list = []
    for batch_idx, (noise_input_batch, image_gt_batch) in enumerate(face_loader_eta):
        if batch_idx < 1:
            noise_input_batch = noise_input_batch.to(device)
            for eta in etas :
                diffusion = Diffusion(
                num_timesteps   =diffusion_config["num_timesteps"], 
                beta_start      =diffusion_config["beta_start"], 
                beta_end        =diffusion_config["beta_end"], 
                device          =device,
                total_steps     =diffusion_config["total_steps"],
                batch_size      =diffusion_config["batch_size"],
                eta             =eta
                )
                generated_images_batch, all_images = diffusion.ddim_sample(model, noise_input_batch)
                eta_images_list.append(generated_images_batch.cpu())
            del diffusion
    save_images_grid_eta(eta_images_list, etas, save_path)
    return 0

def main_ddim(config_path):
    use_gpu     = 1
    device      = "cuda" if torch.cuda.is_available() and use_gpu  else "cpu"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load config parameters
    dataset_config  = config["P2_dataset_params"]
    diffusion_config= config["P2_diffusion_params"]

    noise_folder    = dataset_config["noise_folder"]
    gt_folder       = dataset_config["GT_folder"]
    output_folder   = dataset_config["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = UNet().to(device)

    ckpt_path = dataset_config["ckpt_name"]
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loading checkpoint from {ckpt_path} success!")
    else:
        print(f"Checkpoint {ckpt_path} not found. Exiting.")
        return

    face_dataset    = FaceDataset(dataset_config["noise_folder"], dataset_config["GT_folder"])
    face_loader     = DataLoader(face_dataset, batch_size = diffusion_config["batch_size"], shuffle = False, num_workers = 4)

    diffusion_main      = Diffusion(
        num_timesteps   =diffusion_config["num_timesteps"], 
        beta_start      =diffusion_config["beta_start"], 
        beta_end        =diffusion_config["beta_end"], 
        device          =device,
        total_steps     =diffusion_config["total_steps"],
        batch_size      =diffusion_config["batch_size"],
        eta             =diffusion_config["eta"]
    )

    # Evaluate the model using DDIM and save results
    sample_start_time   = time.time()
    timestep_gird_path  = "./Result/P2_DDIM/ddim_timestep_grid.png"
    avg_mse             = evaluate_ddim_model(
        diffusion_main, 
        model, 
        face_loader,
        save_dir        = dataset_config["output_folder"],
        device          = device,
        timestep_save_path = timestep_gird_path
    )

    print(f"Average MSE across all samples: {avg_mse:.5f}")
    print(f"Sample time: {time.time() - sample_start_time:.3f}")
    
    # Evaluate the model using DDIM different etas
    eta_list        = [0.0,0.25,0.50,0.75,1.0]
    eta_figure_path = './Result/P2_DDIM/eta_images.png'
    face_loader_eta = DataLoader(face_dataset, batch_size = 4, shuffle = False, num_workers = 4)
    vary_eta_generate(model, face_loader_eta, diffusion_config, device, save_path = eta_figure_path, etas = eta_list)

    # Evaluate Noise Interpolation
    noise_0, _ = face_dataset[0] 
    noise_1, _ = face_dataset[1] 
    slerp_path = './Result/P2_DDIM/interpolated_slerp.png'
    linear_path = './Result/P2_DDIM/interpolated_linear.png'

    interpolator = Interpolator(model=model, diffusion=diffusion_main, device=device)
    interpolator.generate_and_save(noise_0, noise_1, steps=11, save_paths=(slerp_path, linear_path))


if __name__ == "__main__":
    config_path = "./MY_UTILS/config.yaml"
    main_ddim(config_path)
