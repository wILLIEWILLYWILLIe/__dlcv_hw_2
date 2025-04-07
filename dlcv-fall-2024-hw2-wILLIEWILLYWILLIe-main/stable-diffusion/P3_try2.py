import os
import time
import numpy as np
import logging
from PIL import Image
import random
from omegaconf import OmegaConf
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def load_model_from_config(config, ckpt, verbose=False):
    """Load model from checkpoint"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:", m)
    if len(u) > 0 and verbose:
        print("unexpected keys:", u)
    model.cuda()
    model.eval()
    return model

class CustomDataset(Dataset):
    def __init__(self, img_dir, token, imgz = 512):
        self.img_dir = img_dir
        self.image_filenames = [
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        if token == '<new2>':
            self.transform = transforms.Compose([
                transforms.Resize(imgz), # Resize the shorter side to 1024 while keeping the aspect ratio
                transforms.RandomCrop((imgz, imgz)),  
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((imgz,imgz)),  
                # transforms.CenterCrop((imgz, imgz)),
                transforms.Resize(imgz),
                transforms.RandomCrop((imgz, imgz)),  
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]) 
            ])
        self.token = token

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path    = self.image_filenames[idx]
        image       = Image.open(img_path).convert("RGB")
        image       = self.transform(image)
        return image, self.token

class TextualInversion:
    def __init__(self, config_path, model_path, special_tokens, device="cuda", train_config = {}, inf_config = {}):
        self.device     = device
        self.config     = OmegaConf.load(config_path)
        self.model      = load_model_from_config(self.config, model_path).to(device)
        self.multi      = 0
        if self.multi:
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])

        self.criterion      = nn.MSELoss()
        self.train_config   = train_config
        self.inf_config     = inf_config
        self.scaler         = None
        if train_config["precision"] == "autocast":
            self.scaler     = torch.cuda.amp.GradScaler()

        # Get text encoder (CLIP) from conditioning stage
        if not self.multi:
            self.img_encoder        = self.model.first_stage_model
            self.text_encoder       = self.model.cond_stage_model            #FrozenCLIPEmbedder
            self.tokenizer          = self.text_encoder.tokenizer       #FrozenCLIPEmbedder-CLIPTokenizer
            self.transformer        = self.text_encoder.transformer     #FrozenCLIPEmbedder-CLIPTextModel
            self.unet               = self.model.model
        else:
            self.img_encoder    = self.model.module.first_stage_model.to("cuda:1")
            self.text_encoder   = self.model.module.cond_stage_model.to("cuda:1")
            self.tokenizer      = self.text_encoder.tokenizer
            self.transformer    = self.text_encoder.transformer
            self.unet           = self.model.module.model.to("cuda:1")
            self.model          = self.model.module.to("cuda:0")

        # Freeze all models
        self.model.eval()
        self.img_encoder.eval()
        self.text_encoder.eval()
        self.transformer.eval()
        self.unet.eval()
        if 1:
            self.model.requires_grad_(False)
            self.img_encoder.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.transformer.requires_grad_(True)
            self.unet.requires_grad_(True)

        # Initialize placeholder token embeddings
        self.token_embeddings   = nn.ParameterDict()
        self.initial_token_embeddings = {}
        self.placeholder_tokens = special_tokens
        self.__initialize_tokens()

        # Inf config setup
        self.output_dir = inf_config["output_folder"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.shape      = [inf_config["latent_channels"], 
                    inf_config["height"] // inf_config["downsample_factor"], 
                    inf_config["width"] // inf_config["downsample_factor"]]
        self.sampler    = DPMSolverSampler(self.model)

        self.original_embeddings = self.transformer.get_input_embeddings().weight.data.clone()
        self.original_unet_params= {
            name: param.detach().clone()
            for name, param in self.unet.named_parameters()
        }

        self.text_prompts = [
        "a photograph of an astronaut riding a horse",
        "A photo of <new1>.",
        "A <new1> shepherd posing proudly on a hilltop with Mount Fuji in the background.",
        "A <new1> perched on a park bench with the Colosseum looming behind.",
        "A photo of <new2>.",
        "The streets of Paris in the style of <new2>.",
        "Manhattan skyline in the style of <new2>."
        ]

    def __initialize_tokens(self):
        logger.debug([[text, '-->', self.tokenizer(text, truncation=True, max_length=self.text_encoder.max_length, 
                    return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")] 
                  for text in self.placeholder_tokens])

        # Add new tokens to tokenizer
        added_tokens = self.tokenizer.add_tokens(self.placeholder_tokens)
        if added_tokens > 0:
            self.transformer.resize_token_embeddings(len(self.tokenizer))
       
        logger.debug([[text, '-->', self.tokenizer(text, truncation=True, max_length=self.text_encoder.max_length, 
                    return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")] 
                  for text in self.placeholder_tokens])

        # Initialize token embeddings as trainable parameters
        token_emb   = self.transformer.get_input_embeddings()
        similar_tokens ={   
            # '<new1>': ["dog", "puppy", "small dog", "cute dog", "Corgi dog"],
            # '<new2>': ["fantasy", "illustration", "painting", "artwork"]
            '<new1>': ["dog"],
            '<new2>': ["imaginative digital paint"],
            '<new3>': ["dog"]
        }
        for token in self.placeholder_tokens:
            token_similar_tokens    = similar_tokens[token]
            similar_token_ids       = self.tokenizer.convert_tokens_to_ids(token_similar_tokens)
            mean_embedding          = token_emb.weight[similar_token_ids].mean(dim=0)

            token_id = self.tokenizer.convert_tokens_to_ids(token)
            logger.info(f'Initialize token {token} --> token ID {token_id}')
            initial_embedding = mean_embedding.clone() + 0.001 * torch.randn_like(mean_embedding)
            self.token_embeddings[str(token_id)] = nn.Parameter(
                # token_emb.weight[token_id].clone().detach()
                initial_embedding.clone(),
                requires_grad=True
            )
            self.initial_token_embeddings[str(token_id)] = initial_embedding.clone()
            with torch.no_grad():
                for token_id, embedding in self.token_embeddings.items():
                    self.transformer.get_input_embeddings().weight.data[int(token_id)] = embedding

    def __get_text_embedding(self, prompts):
        
        batch_encoding  = self.tokenizer(prompts, truncation=True, max_length=self.text_encoder.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        logger.debug(f'GetTextEmbedding: BatchEncoded {batch_encoding["input_ids"]}')

        # Original Way
        #################################################
        # tokens = batch_encoding["input_ids"].to(self.device)
        # outputs = self.transformer(input_ids=tokens)
        #################################################

        # New Way 1
        #################################################
        # tokens      = batch_encoding["input_ids"].to(self.device).long()
        # # Before the forward pass, temporarily replace the embeddings
        # orig_embeds = self.transformer.get_input_embeddings()
        # orig_weight = orig_embeds.weight.data.clone()
        # # Replace the embeddings for our special tokens
        # with torch.no_grad():
        #     for token in self.placeholder_tokens:
        #         token_id = self.tokenizer.convert_tokens_to_ids(token)
        #         orig_embeds.weight.data[token_id] = self.token_embeddings[str(token_id)]
        # # Forward pass
        # outputs = self.transformer(input_ids=tokens)
        # # Restore the original embeddings
        # with torch.no_grad():
        #     orig_embeds.weight.data = orig_weight
        #################################################

        # New Way 2
        #################################################
        tokens      = batch_encoding["input_ids"].to(self.device).long()
        with torch.no_grad():
            for token_id, token_embedding in self.token_embeddings.items():
                token_id = int(token_id)
                self.transformer.get_input_embeddings().weight.data[token_id] = token_embedding
        outputs = self.transformer(input_ids=tokens)
        #################################################
        z = outputs.last_hidden_state
        return z

    def train_tokens(self, dataloader):

        best_loss           = float('inf')
        train_start         = time.time()
        prompt_templates = [
            "{}.",
            "A photo of {}.",
            "An artistic depiction of {}.",
            "A sketch of {}.",
            "An illustration of {}.",
            "A view of {}.",
            "A scene showing {}.",
            "One figure of {}.",
            "{} image.",
            "Style of {}.",
            "Impression of {}.",
            "Image of {}.",
            "Photo in a {} style.",
            "A portrait of {}."
        ]

        training_initial_embeddings = {
            token_id: param.clone().detach() 
            for token_id, param in self.token_embeddings.items()
        }

        for name, param in self.token_embeddings.items():
            logger.info(f"train_tokens: self.token_embeddings{name}: requires_grad={param.requires_grad}")
        self.transformer.get_input_embeddings().requires_grad_(True)
        embedding_params    = self.transformer.get_input_embeddings().parameters()
        self.optimizer      = torch.optim.AdamW(embedding_params, lr=self.train_config["learning_rate"])


        for epoch in range(self.train_config["num_epochs"]):
            epoch_loss      = 0
            epoch_start_embeddings = {
                token_id: param.clone().detach() 
                for token_id, param in self.token_embeddings.items()
            }
            for idx, (images, tokens) in enumerate(dataloader):
                images  = images.to(self.device)
                random_prompt_template1 = random.choice(prompt_templates)
                random_prompt_template2 = random.choice(prompt_templates)
                loss    = self.train_step(images, tokens, prompt_template = random_prompt_template1)
                epoch_loss += loss
                loss    = self.train_step(images, tokens, prompt_template = random_prompt_template2)
                epoch_loss += loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            elapsed_time = time.time() - train_start
            mins, secs = divmod(elapsed_time, 60)
    
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.train_config['num_epochs']}, Loss: {avg_epoch_loss:.3f}, Time: {int(mins)} min {int(secs)} sec")
            if epoch % 50 == 0:
                self.__check_epoch_progress(training_initial_embeddings, epoch_start_embeddings, epoch)

            # Save checkpoint if loss improves
            # if epoch in [0,5,10,50,75,100] or epoch % 100 == 0 :
            if epoch in [0, 5, 10, 50, 75, 100] or epoch % 100 == 0 or (400 <= epoch <= 900 and (epoch - 400) % 20 == 0):
                self.save_checkpoint(epoch, best_loss, save_path=f'./ckpt_folder_new/epoch{epoch}_2_ckpt')
                # torch.cuda.empty_cache()
                # with torch.no_grad():
                #     for idx, p in enumerate(self.text_prompts):
                #         ti.inference(prompt_template=p, num_iter = 1, save_path = f'sample_epoch{epoch}_p{idx}', save_num = 1, low_reslou = True)

        self.save_checkpoint(epoch, avg_epoch_loss, save_path=f'last_ckpt')

    def train_step(self, images, tokens, prompt_template = "A photo of {}."):

        self.optimizer.zero_grad()
        initial_embeddings = {
            token_id: param.clone().detach() 
            for token_id, param in self.token_embeddings.items()
        }

        # Generate embeddings for the prompt text
        prompts = [prompt_template.format(token) for token in tokens]
        logger.debug(f'train_step--trian_prompts: {prompts}')
        text_input_ids = self.__get_text_embedding(prompts)

        # with torch.set_grad_enabled(True):
        with torch.no_grad():
            # Encode images to latents and perform diffusion step
            latents = self.model.encode_first_stage(images)
            latents = self.model.get_first_stage_encoding(latents)

        noise           = torch.randn_like(latents, requires_grad=True)
        timesteps       = torch.randint(0, self.model.num_timesteps, (images.shape[0],), device=device)
        noisy_latents   = self.model.q_sample(latents, timesteps, noise)
        # noisy_latents   = noisy_latents.requires_grad_(True)

        # Compute loss with respect to custom embeddings
        pred_noise  = self.model.apply_model(noisy_latents, timesteps, text_input_ids)
        loss        = self.criterion(pred_noise, noise)

        logger.debug(f"Pred noise requires_grad: {pred_noise.requires_grad}")
        logger.debug(f"Latents requires_grad: {latents.requires_grad}")
        logger.debug(f"Noisy Latents requires_grad: {noisy_latents.requires_grad}")
        logger.debug(f"Text embeddings requires_grad: {text_input_ids.requires_grad}")

        # Backpropagation
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()  
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([param for param in self.token_embeddings.values()], 1.0)

        # Optimization step
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            current_embeddings  = self.transformer.get_input_embeddings().weight.data
            mask                = torch.ones_like(current_embeddings, dtype=torch.bool)
            pre_update_embeddings = {
                token_id: param.clone().detach() 
                for token_id, param in self.token_embeddings.items()
            }
            for token_id in self.token_embeddings.keys():
                token_id_int = int(token_id)
                # Update our stored token embeddings with the latest values
                self.token_embeddings[token_id] = nn.Parameter(
                    current_embeddings[token_id_int].clone(),
                    requires_grad=True
                )
                mask[token_id_int] = False
            # Restore original embeddings except for new tokens
            current_embeddings[mask] = self.original_embeddings[mask]
            for token_id, embedding in self.token_embeddings.items():
                current_embeddings[int(token_id)] = embedding

            # Update transformer embeddings
            self.transformer.get_input_embeddings().weight.data.copy_(current_embeddings)
            
            # Restore original UNet parameters
            for name, param in self.unet.named_parameters():
                param.copy_(self.original_unet_params[name])

            # self.__check_embedding_updates(initial_embeddings, pre_update_embeddings)

        return loss.item()
        
    def __check_embedding_updates(self, initial_embeddings, pre_update_embeddings):
        """Check if token embeddings have been updated during the training step"""
        for token_id, current_param in self.token_embeddings.items():
            initial_param       = initial_embeddings[token_id]
            pre_update_param    = pre_update_embeddings[token_id]
            current_param_data  = current_param.data

            # Calculate differences
            diff_from_initial = torch.norm(current_param_data - initial_param).item()
            diff_from_pre_update = torch.norm(current_param_data - pre_update_param).item()
            
            logger.info(f"\nToken {token_id} embedding updates:")
            logger.info(f"  Change from initial: {diff_from_initial:.6f}")
            logger.info(f"  Change from pre-update: {diff_from_pre_update:.6f}")
            
            if diff_from_initial < 1e-6:
                logger.warning(f"  Warning: Token {token_id} shows very small change from initial state")
            if diff_from_pre_update < 1e-6:
                logger.warning(f"  Warning: Token {token_id} shows very small change during update")

            # Optional: Print some actual values for detailed inspection
            logger.debug(f"  Initial values (first 5): {initial_param[:5]}")
            logger.debug(f"  Current values (first 5): {current_param_data[:5]}")

    def __check_epoch_progress(self, training_initial_embeddings, epoch_start_embeddings, epoch):
        """Check embedding changes throughout the training process"""
        print(f"{'-'*20}\nEmbedding Change Analysis (Epoch {epoch}):")
        for token_id, current_param in self.token_embeddings.items():
            training_initial = training_initial_embeddings[token_id]
            epoch_initial = epoch_start_embeddings[token_id]
            current = current_param.data

            total_change = torch.norm(current - training_initial).item()
            epoch_change = torch.norm(current - epoch_initial).item()

            print(f"Token {token_id}:")
            print(f"  Total change since training start: {total_change:.6f}")
            print(f"  Change since epoch start: {epoch_change:.6f}")
        print(f"{'-'*20}")

    def save_checkpoint(self, epoch, loss, save_path="learned_ckpt"):
        checkpoint_path = os.path.join(f"{save_path}.pth")
        if epoch == 0:
            torch.save({
                "epoch": epoch,
                "token_embeddings": self.token_embeddings,
            "initial_token_embeddings": self.initial_token_embeddings
                # "loss": loss
            }, checkpoint_path)
        else:
            torch.save({
                "epoch": epoch,
                "token_embeddings": self.token_embeddings,
                # "loss": loss
            }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path ,token_ids_to_update=None):
        if self.multi:
            checkpoint              = torch.load(checkpoint_path, map_location='cuda:1')
        else:
            checkpoint              = torch.load(checkpoint_path, map_location=self.device)
        # self.model.load_state_dict(checkpoint["model_state_dict"])
        # self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint_token_embeddings       = checkpoint["token_embeddings"]

        if checkpoint.get("epoch", 0) == 0:
            if "initial_token_embeddings" in checkpoint:
                for token_id, initial_embedding in checkpoint["initial_token_embeddings"].items():
                    self.token_embeddings[token_id] =  nn.Parameter(initial_embedding.clone())

        with torch.no_grad():
            if token_ids_to_update is not None:
                for token_id_int in token_ids_to_update:
                    token_id_str = str(token_id_int)  # Convert to string for dictionary matching
                    embedding = checkpoint_token_embeddings.get(token_id_str)
                    
                    if embedding is not None:
                        self.transformer.get_input_embeddings().weight.data[token_id_int] = embedding
                        self.token_embeddings[token_id_str] = embedding
                        logger.info(f"Token ID {token_id_int} updated.")
                    else:
                        logger.warning(f"Token ID {token_id_int} not found in token_embeddings.")
            else:
                self.token_embeddings       = checkpoint["token_embeddings"]
                for token_id, embedding in self.token_embeddings.items():
                    self.transformer.get_input_embeddings().weight.data[int(token_id)] = embedding
                logger.info(f"All token embeddings updated: {list(self.token_embeddings.keys())}")
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def inference(self, prompt_template="A photo of <new1>.",
                         num_iter = 1, 
                         save_path = 'inf', 
                         save_num = 1, 
                         save_dir = None,
                         low_reslou = False,
                         ddimstep = None,
                         eta = None,
                         scale = None
                         ):
        """ Run inference after loading the model and embeddings """
        if save_dir is None:
            save_dir = self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        self.model.eval()
        inf_batch_size  = self.inf_config.get("inf_batch_size", 1)
        prompts_data    = [[prompt_template] * inf_batch_size]
        prompts_data    = [prompt_template]
        
        base_count      = 0

        start_code      = None
        _ddimstep   = self.inf_config["ddim_steps"] if ddimstep is None  else ddimstep
        _eta        = self.inf_config["ddim_eta"] if eta is None else eta
        _scale      = self.inf_config["guidance_scale"] if scale is None else scale

        if low_reslou:
            shape = self.half_shape
        else: shape = self.shape
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for n in range(num_iter):
                    # for prompts in prompts_data:
                    if True:
                        # print(f'inf prompt: {prompts_data}')
                        uc = None
                        if self.inf_config["guidance_scale"] != 1.0:
                            uc = self.model.get_learned_conditioning(inf_batch_size * [""])
                        c = self.__get_text_embedding(prompts_data)
                        samples_ddim, _ = self.sampler.sample(S=_ddimstep,
                                                            conditioning=c,
                                                            batch_size=inf_batch_size,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=_scale,
                                                            unconditional_conditioning=uc,
                                                            eta=_eta,
                                                            x_T=start_code)
                        
                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img_save_path = os.path.join(save_dir, f"{save_path}_{base_count+1}.png")
                            img.save(img_save_path)
                            print(f"Saved image to {img_save_path}", end = '\r')
                            if base_count == save_num-1:
                                print(f'\nInf {save_path} done')
                                return
                            base_count += 1
        print(f'\nInf {save_path} done')
        return 
                        

ti_config = {
        "tokens_to_train": ["<new1>", "<new2>", "<new3>"],
        "config_path"   : "./configs/stable-diffusion/v1-inference.yaml",
        "model_path"    : "./ldm/models/stable-diffusion-v1/model.ckpt",
        "new1_img_dir"  : "../hw2_data/textual_inversion/0",
        "new2_img_dir"  : "../hw2_data/textual_inversion/1",
        "new3_img_dir"  : "../hw2_data/textual_inversion/cat",
        "new4_img_dir"  : "../hw2_data/textual_inversion/mydog",
        "do_train"      : 1,
        "train_config"  :{
            "train_batch_size"  : 2,
            "num_epochs"        : 3000,
            "trian_imgz"        : 256,
            "learning_rate"     : 5e-4,
            "precision"         : "autocast",
        },
        "do_sample"     : 0 ,
        "sample_config" :{
            "inf_batch_size"    : 1,
            "output_folder"     : "./output",
            "latent_channels"   : 4,
            "downsample_factor" : 8,
            "height"            : 512,
            "width"             : 512,
            "guidance_scale"    : 6.5, #unconditional guidance scale
            "ddim_steps"        : 70,
            "ddim_eta"          : 0.1
        }
    }

if __name__ == "__main__":
    config = ti_config

    # tokens_to_train = ['dog',"<new1>", "<new2>"]
    device          = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    ti              = TextualInversion( config["config_path"], 
                                        config["model_path"], 
                                        config["tokens_to_train"],
                                        device=device, 
                                        train_config=config["train_config"], 
                                        inf_config = config["sample_config"]
                                    )
    dataset1        = CustomDataset(config["new1_img_dir"], token=config["tokens_to_train"][0], imgz = config["train_config"]["trian_imgz"])
    dataset2        = CustomDataset(config["new2_img_dir"], token=config["tokens_to_train"][1], imgz = config["train_config"]["trian_imgz"])
    dataset3        = CustomDataset(config["new4_img_dir"], token=config["tokens_to_train"][2], imgz = config["train_config"]["trian_imgz"])
    # combined_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    # dataloader      = DataLoader(combined_dataset, batch_size=config["train_config"]["train_batch_size"], shuffle=True, num_workers = 4)
    dataloader      = DataLoader(dataset2, batch_size=config["train_config"]["train_batch_size"], shuffle=True, num_workers = 4)
    

    if config["do_train"]:
        # ti.load_checkpoint(f'./epoch600new_ckpt.pth')
        ti.train_tokens(dataloader=dataloader)

    test_prompts = [
        # "a photograph of an astronaut riding a horse",
        "A photo of <new1>.",
        "A <new1> shepherd posing proudly on a hilltop with Mount Fuji in the background.",
        "A <new1> perched on a park bench with the Colosseum looming behind.",
        "A photo of <new2>.",
        "The streets of Paris in the style of <new2>.",
        "Manhattan skyline in the style of <new2>.",
        "A photo of <new3>.",
        "<new3> in the style of <new2>."
    ]

    if config["do_sample"]:
        ti.load_checkpoint(f'./epoch{epoch}_3_ckpt.pth',token_ids_to_update=[49410])
        for idx, p in enumerate(test_prompts):
            if idx in [6,7]:
                ti.inference(prompt_template=p, num_iter = 1, save_path = f'epoch{epoch}_p{idx}', save_num = 1)


