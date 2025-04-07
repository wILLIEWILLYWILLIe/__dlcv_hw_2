import os
import random
import yaml
import time
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
from torchvision.utils import save_image ,make_grid
from torchvision import transforms
# from P1_dataloader import DigitsDataset_list
# from P1_model import UNet_conditional
# from P1_Unet import UNet
from MY_UTILS.P1_dataloader import DigitsDataset_list
from MY_UTILS.P1_model import UNet_conditional
from MY_UTILS.P1_Unet import UNet

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)  
    return config

def save_images_train(images, path, **kwargs):
    grid = make_grid(images, **kwargs)
    ndarr = grid.permute(1,2,0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def save_images_grid(sampled_images, labels, save_path, img_size=(32, 32)):
    """
    Save the sampled images as a grid, with one row for each label.
    :param sampled_images: Tensor of generated images.
    :param labels: Tensor of corresponding labels.
    :param save_path: Path to save the grid image.
    :param img_size: Size of each image in the grid.
    """
    unique_labels = torch.unique(labels)
    images_per_label = len(sampled_images) // len(unique_labels)
    # print(f"Unique labels: {unique_labels}, Images per label: {images_per_label}")

    rows = []
    for label in unique_labels:
        label_images = sampled_images[labels == label].cpu()

        # Make sure the pixel values are in [0, 1]
        label_images = (label_images - label_images.min()) / (label_images.max() - label_images.min())
        # Concatenate horizontally for each label
        label_row = torch.cat([img.permute(1, 2, 0) for img in label_images], dim=1)
        rows.append(label_row)

    grid = torch.cat(rows, dim=0).numpy()
    grid_image = Image.fromarray((grid * 255).astype(np.uint8))
    grid_image.save(save_path)
    print(f"Saved image grid to {save_path}")

def sample_images_mode( diffusion, model, device = 'cuda', sample_num = 50, mode = 'grid',
                        dataset_name = 'mnistm', save_path = '', sample_once = 0):

    print(f"{'-'*50}\nSampling image and save in folder {save_path}")
    # if not os.path.exists(save_path):
    #     print(f"Save folder does not exist. Creating folder: {save_path}\n{'-'*50}")
    #     os.makedirs(save_path)
    # else:
    #     print(f"Save folder exists: {save_path}\n{'-'*50}")

    if dataset_name == 'mnistm':
        labels_range = torch.arange(0, 10).long().to(device)  # MNIST-M labels 0-9
        cfg_scale = 3
    elif dataset_name == 'svhn':
        labels_range = torch.arange(10, 20).long().to(device)  # SVHN labels 10-19
        cfg_scale = 3
    else:
        raise ValueError("Unknown dataset name, expected 'mnistm' or 'svhn'.")

    if not sample_once:
        # Using For Loop For Sampling, Slower but uses less memory
        sampled_images = []
        sampled_labels = []
        for label in labels_range:
            print('inf label ', label)
            labels = torch.full((sample_num,), label, device=device).long()  
            label_images = diffusion.sample(model, n=sample_num, labels=labels, cfg_scale=cfg_scale)
            sampled_images.append(label_images)
            sampled_labels.append(labels)
        # Stack all images and labels together
        all_labels      = torch.cat(sampled_labels, dim=0)  # Shape: (100,)
        sampled_images  = torch.cat(sampled_images, dim=0)  # Shape: (100, channels, height, width)
    
    else:
        # Sample all images at once, run diffusion.sample only once
        all_labels      = torch.cat([torch.full((sample_num,), label, device=device).long() for label in labels_range])
        sampled_images  = diffusion.sample(model, n=len(all_labels), labels=all_labels, cfg_scale=cfg_scale)

    if mode == 'inf':
        for idx, label in enumerate(labels_range):
            label_images = sampled_images[idx * sample_num:(idx + 1) * sample_num]  
            for i, img in enumerate(label_images):
                filename = f"{label%10}_{i+1:03}.png"  # Label % 10 ensures labels go from 0-9 for both MNISTM and SVHN
                image_save_path = os.path.join(save_path, filename)
                # Convert to numpy and save
                grid = make_grid(img)
                ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
                im = Image.fromarray(ndarr)
                im.save(image_save_path)
                print(image_save_path)
        print("Images saved individually in 'inf' mode.")

    elif mode == 'grid':
        print(f"Sampling {sample_num} images for each label for {dataset_name} and saving in a grid format.")
        zero_visualization_path = os.path.join(save_path, f"{dataset_name}_zero_visualization.png")
        if dataset_name == 'mnistm':
            x_mnistm, x_svhn = visualize_both_top_left_zeros(diffusion, model, device,mnistm_save_path = zero_visualization_path)
            print(sampled_images[0].shape, x_mnistm.shape)
            sampled_images[0] = x_mnistm.squeeze(0) 
        elif dataset_name == 'svhn':
            x_mnistm, x_svhn = visualize_both_top_left_zeros(diffusion, model, device,svhn_save_path = zero_visualization_path)
            print(sampled_images[0].shape, x_svhn.shape)
            sampled_images[0] = x_svhn.squeeze(0) 
        grid_save_path          = os.path.join(save_path, f"{dataset_name}_grid.png")
        save_images_grid(sampled_images, all_labels, grid_save_path)
        print("Images saved as a grid in 'grid' mode.")

    else:
        raise ValueError("Unknown mode. Use 'inf' for individual images or 'grid' for grid mode.")

    return 0

def visualize_both_top_left_zeros(diffusion, model, device, mnistm_save_path=None, svhn_save_path=None):
    TIMESTEPS=[0, 200, 400, 600, 800, 999]
    TIMESTEPS=[0, 50, 100, 200, 500, 999]
    mnist_images, svhn_images = [],[]
    x_mnistm, x_svhn = None, None
    # MNIST-M '0'
    if mnistm_save_path is not None:
        print("Visualizing MNIST-M '0' (top-left) reverse process...")
        x_mnistm = diffusion.visualize_timesteps(model, label=0, timesteps=TIMESTEPS, save_path=mnistm_save_path)

    # SVHN '0' (label 10)
    if svhn_save_path is not None:
        print("Visualizing SVHN '0' (top-left) reverse process...")
        x_svhn = diffusion.visualize_timesteps(model, label=10, timesteps=TIMESTEPS, save_path=svhn_save_path)
    return x_mnistm, x_svhn

class Diffusion:

    def __init__(self, num_timesteps=1000, beta_start = 1e-4, beta_end = 0.02, imgz = 32, device = "cuda"):
        self.num_timesteps  = num_timesteps
        self.beta_start     = beta_start
        self.beta_end       = beta_end
        self.imgz           = imgz
        self.device         = device

        self.betas      = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas     = 1. - self.betas
        self.alpha_hat  = torch.cumprod(self.alphas, dim = 0)

    def noise_images(self, x, t):
        sqrt_alpha_hat  = torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:,None,None,None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low = 1, high = self.num_timesteps, size = (n,))
    
    def sample(self, model, n, labels, cfg_scale=3):
        r"""Takes in the noise image and the timestep, returns generated images"""
        print(f"-> Sampling {n} images", end = '\r')
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.imgz, self.imgz)).to(self.device)
            for i in reversed(range(0, self.num_timesteps)):
                print(f'DDPM sample timestep {i}', end = '\r')
                t = (torch.ones(n) * i).long().to(self.device) # The current timestep for all `n` images
                # --> Runs over all timesteps(from last to first), less noisy than previous
                # print(f"Shape: {x.shape}, Dtype: {x.dtype}, t_Dtype: {t.dtype}")
                predicted_noise = model(x,t, labels)

                if cfg_scale > 0 :
                    uncond_predicted_noise = model(x, t, None)
                    # Linear interpolates between uncond and condi, balance sampling
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha       = self.alphas[t][:,None, None, None]
                alpha_hat   = self.alpha_hat[t][:,None, None, None]
                beta        = self.betas[t][:,None, None, None]

                if i > 1:
                    noise = torch.randn_like(x) # Add new noise, except for the last step
                else:
                    noise = torch.zeros_like(x) # No noise for the last step
                # Update image x by substract the preticted noise from it
                x = 1/torch.sqrt(alpha) * (x-((1-alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1,1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def visualize_timesteps(self, model, label, timesteps = [0, 200, 400, 600, 800, 1000], save_path = '', cfg_scale=3):
        # print(f"--> Visualizing label {label} across timesteps {timesteps}")
        images_at_timesteps = []
        model.eval()

        with torch.no_grad():
            x = torch.randn((1, 3,self.imgz, self.imgz)).to(self.device)
            label_tensor = torch.tensor(label).long().to(self.device)
            for i in reversed(range(0, self.num_timesteps)):
                t = (torch.ones(1) * i).long().to(self.device)
                predicted_noise = model(x,t,label_tensor)

                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    # Linear interpolates between uncond and condi, balance sampling
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha       = self.alphas[t][:,None, None, None]
                alpha_hat   = self.alpha_hat[t][:,None, None, None]
                beta        = self.betas[t][:,None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
                
                if i in timesteps:
                    step_image = x.clone().clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy() 
                    images_at_timesteps.append(step_image)
        
        grid = np.concatenate(images_at_timesteps, axis=1)  

        grid_image = Image.fromarray((grid * 255).astype(np.uint8))  
        grid_image.save(save_path)

        x = (x.clamp(-1,1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

        # grid = make_grid(torch.cat(images_at_timesteps, dim=0), nrow=len(timesteps))
        # ndarr = grid.permute(1, 2, 0).cpu().numpy()
        # img = Image.fromarray((ndarr * 255).astype(np.uint8))
        # img.save(save_path)
        # print(f"Saved visualization at {save_path}")
                

def main(config_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config          = load_config(config_path)
    dataset_config  = config["P1_dataset_params"]
    model_config    = config["P1_model_params"]
    train_config    = config["P1_train_params"]
    diffusion_config= config["P1_diffusion_params"]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(degrees=5),  
        transforms.ToTensor(), 
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    ])

    P1_dataset  = DigitsDataset_list(
        csv_paths = [dataset_config["MNISTM_CSV"], dataset_config["SVHN_CSV"]],
        dir_paths = [dataset_config["MNISTM_DATA_DIR"], dataset_config["SVHN_DATA_DIR"]],
        transform = transform
    )

    P1_loader   = DataLoader(P1_dataset, batch_size = train_config["batch_size"], shuffle = True, num_workers = 16)

    diffusion   = Diffusion(
        num_timesteps = diffusion_config["num_timesteps"], 
        beta_start = diffusion_config["beta_start"], 
        beta_end = diffusion_config["beta_end"], 
        imgz = model_config["im_size"]
    )

    # model       = UNet_conditional(num_classes = train_config["num_classes"]).to(device)
    model       = UNet(num_classes = train_config["num_classes"]).to(device)

    if train_config["load_ckpt"] and os.path.exists(train_config["load_ckpt_name"]):
        print(f"Loading checkpoint from {train_config['load_ckpt_name']}\n", end = '\r')
        try:
            checkpoint = torch.load(train_config["load_ckpt_name"], map_location=device)
            model.load_state_dict(checkpoint)
            print("--> Checkpoint loaded successfully.")
        except Exception as e:
            print(f"--> Error loading checkpoint: {e}")
    
    if train_config["sample_only"]:
        model.eval()
        # sampling_time = time.time()
        # for dataset_name, save_path in [
        #                                 ['mnistm', dataset_config["MNISTM_OUTPUT_DIR"]], 
        #                                 ['svhn', dataset_config["SVHN_OUTPUT_DIR"]]
        #                                 ]:
        #     sample_images_mode( diffusion, 
        #                         model, 
        #                         device, 
        #                         sample_num = train_config["num_samples"], 
        #                         mode = 'inf',
        #                         dataset_name = dataset_name,
        #                         save_path = save_path
        #     )
        # print(f'Using {time.time() - sampling_time:.3f} to sample images')
        for dataset_name in ["mnistm", "svhn"]:
            sample_images_mode( diffusion, 
                                model, 
                                device, 
                                sample_num  = 10, 
                                mode        = 'grid',
                                save_path   = os.path.join('Result', train_config["task_name"]), 
                                dataset_name= dataset_name,
                                sample_once = 1
            )
        # visualize_both_top_left_zeros(diffusion, model, device,
        #         mnistm_save_path    = os.path.join('Result', train_config["task_name"], "mnistm_zero_visualization.png"),
        #         svhn_save_path      = os.path.join('Result', train_config["task_name"], "svhn_zero_visualization.png")
        # )
        return

    optimizer   = torch.optim.AdamW(model.parameters(), lr = train_config["lr"])
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    mse_loss    = nn.MSELoss()

    # discriminator           = Discriminator(img_size=32, channels=3).to(device)
    # discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.0002)
    # adversarial_loss        = nn.BCEWithLogitsLoss() 

    # Start training
    for epoch in range(train_config["num_epochs"]):
        epoch_start = time.time()
        epoch_loss  = 0
        for batch_idx, (image_path, images, labels, dataset_idx) in enumerate(P1_loader):
            if True:
            # if batch_idx < 10:
                sys.stdout.write(f"\r[{epoch + 1}/{train_config['num_epochs']}] Train batch: {batch_idx+1} / {len(P1_loader)} --epoch time: {time.time()-epoch_start:.3f}")
                sys.stdout.flush()

                labels = (10*dataset_idx + labels).to(device)
                images = images.to(device)
                # print(images.shape)

                # Sample timesteps and add noise to images
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)

                # Set 10% of samples as unconditional
                if np.random.random() < 0.1:
                    labels = None

                # Predict noise
                predicted_noise = model(x_t, t, labels)
                loss            = mse_loss(noise, predicted_noise)

                ####################################
                # Train the Discriminator
                ####################################
                # real_labels = torch.ones(images.size(0), 1, device=device)
                # fake_labels = torch.zeros(images.size(0), 1, device=device)

                # discriminator_optimizer.zero_grad()

                # real_loss = adversarial_loss(discriminator(images), real_labels)
                # fake_loss = adversarial_loss(discriminator(x_t.detach()), fake_labels)  # Don't propagate through UNet
                # discriminator_loss = (real_loss + fake_loss) / 2

                # discriminator_loss.backward()
                # discriminator_optimizer.step()

                # # Add adversarial loss to the total loss for UNet
                # total_loss = loss + 0.1 * discriminator_loss.detach()  

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                optimizer.zero_grad()
        scheduler.step()

        torch.save(model.state_dict(), train_config["save_ckpt_name"])
        if epoch % 10 ==0:
            sample_num      = 20
            labels          = torch.arange(sample_num).long().to(device)
            sampled_images  = diffusion.sample(model, n = sample_num, labels=labels)
            save_images_train(sampled_images, os.path.join('Result', train_config["task_name"], f'epoch_{epoch}.jpg'))
        else:
            print(f" -epoch loss: {epoch_loss:.3f}")



if __name__ == "__main__":
    config_path = "./MY_UTILS/config.yaml"
    main(config_path = config_path)
    pass
