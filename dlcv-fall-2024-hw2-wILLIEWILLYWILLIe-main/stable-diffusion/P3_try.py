import os, time
import numpy as np
import logging
from tqdm import tqdm, trange
from einops import rearrange
from omegaconf import OmegaConf
from contextlib import contextmanager, nullcontext
from PIL import Image
import torch
import torch.nn as nn
from torch import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

logging.basicConfig(level=logging.INFO)
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
    def __init__(self, img_dir, token, prompt_template="A photo of {}", imgz = 512):
        self.img_dir = img_dir
        self.image_filenames = [
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize(imgz),
            transforms.CenterCrop(imgz),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ])
        self.token = token
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        prompt = self.prompt_template.format(self.token)
        return image, prompt, self.token

class TextualInversion:
    def __init__(self, config_path, model_path, special_tokens, device="cuda", precision_scope=None):
        self.device     = device
        self.config     = OmegaConf.load(config_path)
        self.model      = load_model_from_config(self.config, model_path).to(device)
        self.criterion  = nn.MSELoss()

        if precision_scope == "autocast":
            self.precision_scope = autocast
        elif precision_scope == "fp16":
            self.precision_scope = torch.cuda.amp.autocast
        else:
            self.precision_scope = nullcontext

        self.special_tokens     = special_tokens
        self.token_embeddings   = {} 

        self.initialize_tokens3()

    def initialize_tokens2(self):
        """Alternative initialization using Xavier initialization."""
        logger.info("Initializing special tokens with Xavier initialization")
        
        # Initialize embedding dimensions
        self.embedding_dim = self.model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.shape[1]

        # Initialize token embeddings using Xavier Initialization
        with torch.no_grad():
            for token in self.special_tokens:
                logger.debug(f"Initializing token: {token}")
                self.token_embeddings[token] = torch.empty(self.embedding_dim).to(self.device)
                nn.init.xavier_normal_(self.token_embeddings[token].unsqueeze(0))  # Xavier init
                self.token_embeddings[token].requires_grad = True
                logger.debug(f"Token '{token}' initialized with Xavier")

        logger.info("Special tokens initialized successfully")
        return 0

    def initialize_tokens3(self):
        # Initialize token embeddings with random small values
        embedding_dim = self.model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.shape[1]
        tokenizer = self.model.cond_stage_model.tokenizer
        embedding_layer = self.model.cond_stage_model.transformer.text_model.embeddings.token_embedding
        similar_words = {
            '<new1>': ['dog', 'puppy', 'animal', 'pet'],  # For animal-like tokens
            '<new2>': ['painting', 'artwork', 'style', 'artistic']  # For style tokens
        }

        for token in self.special_tokens:
            # Get similar word embeddings
            similar_embeds = []
            for word in similar_words.get(token, []):
                word_ids = tokenizer(word, return_tensors="pt").input_ids
                word_embed = embedding_layer.weight[word_ids[0, 1]].detach()  # Skip BOS token
                similar_embeds.append(word_embed)
            
            # Initialize with mean of similar embeddings + small noise
            if similar_embeds:
                init_embed = torch.stack(similar_embeds).mean(0)
                init_embed += torch.randn_like(init_embed) * 0.1
            else:
                init_embed = torch.randn(embedding_dim) * 0.1
            
            self.token_embeddings[token] = nn.Parameter(
                init_embed.unsqueeze(0).to(self.device)
            )

        for token, embedding in self.token_embeddings.items():
            embedding.requires_grad = True
        return 0
    
    def update_model_embedding_layer(self):
        """After training, update the model's internal embedding layer with the trained embeddings"""
        
        token_embeds = self.model.cond_stage_model.transformer.text_model.embeddings.token_embedding
        
        # Update the model's embedding layer with the trained embeddings
        for token in self.special_tokens:
            token_id = self.model.cond_stage_model.tokenizer.convert_tokens_to_ids(token)
            
            with torch.no_grad():
                token_embeds.weight.data[token_id] = self.token_embeddings[token].detach()

        print("Successfully integrated the trained embeddings into the model's embedding layer.")

    def train_tokens(self, dataloader, train_config = None):
        """Train the token embeddings using image comparison with DPM Solver sampling"""

        embedder            = self.model.cond_stage_model.transformer.text_model.embeddings.token_embedding
        embedder.weight.requires_grad = True
        
        self.optimizer      = torch.optim.AdamW(embedder.parameters(), lr=train_config["learning_rate"])
        sampler             = DPMSolverSampler(self.model)
        
        
        train_batch_size=train_config["train_batch_size"]
        shape = [
            train_config["latent_channels"], 
            train_config["height"] // train_config["downsample_factor"], 
            train_config["width"] // train_config["downsample_factor"]
        ]

        self.model.first_stage_model.eval()
        embedder.train()

        print("Initial embedding requires_grad:", embedder.weight.requires_grad)
        print("----------Starting training...----------")

        for epoch in range(train_config["num_epochs"]):
            running_loss = 0.0
            
            for i, (target_images, prompts) in enumerate(dataloader):
                # print(f"Epoch [{epoch+1}/{train_config["num_epochs"]}], Batch [{i}/{len(dataloader)}], Loss: {running_loss:.3f}", end = '\r')
                print(f"Epoch [{epoch+1}/{train_config['num_epochs']}], Batch [{i}/{len(dataloader)}], Loss: {running_loss:.3f}")
                optimizer.zero_grad()
                
                target_images = target_images.to(self.device)
                with torch.no_grad():
                    target_latents = self.model.first_stage_model.encode(target_images)
                    target_latents = target_latents.mode().detach()
                
                # Get conditional embeddings with gradient tracking
                # with torch.set_grad_enabled(True):
                with torch.enable_grad():
                    c = self.model.get_learned_conditioning(prompts)
                    uc = self.model.get_learned_conditioning([""] *train_batch_size)

                    samples_ddim, _ = sampler.sample(
                        S=train_config["ddim_steps"],
                        conditioning=c,
                        batch_size=train_batch_size,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=train_config["guidance_scale"],
                        unconditional_conditioning=uc,
                        eta=train_config["ddim_eta"]
                    )
                    
                    # Decode to image space
                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    # Compute loss
                    x_samples_ddim.requires_grad_(True)
                    target_images.requires_grad_(True)

                    latent_loss = self.criterion(samples_ddim, target_latents)
                    pixel_loss  = self.criterion(x_samples_ddim, target_images)
                    loss        = latent_loss + pixel_loss

                print(f"Embedder grad before backward: {embedder.weight.grad is not None}")
                loss.backward(retain_graph=True)
                print(f"Embedder grad after backward: {embedder.weight.grad is not None}")
                if embedder.weight.grad is not None:
                    print(f"Grad norm: {embedder.weight.grad.norm():.3f}")
                
                # Zero out gradients for non-trainable tokens
                grad_mask = torch.zeros_like(embedder.weight, device=self.device)
                grad_mask[trainable_indices] = 1
                
                # Apply gradient mask - only if gradients exist
                if embedder.weight.grad is not None:
                    embedder.weight.grad = embedder.weight.grad * grad_mask
                else:
                    print("\tWarning: No gradients computed!")
                    print(f"\tLoss value: {loss.item():.3f}")
                    print("\tEmbedder requires_grad:", embedder.weight.requires_grad)
                    print("\tEmbedder.weight.grad: ", embedder.weight.grad)
                    print("\tLoss requires_grad:", loss.requires_grad)
                
                optimizer.step()
                running_loss += loss.item()
                
                # if i % 5 == 0:
                #     print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
                # Clear memory
                torch.cuda.empty_cache()
            
            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], {'-'*10} Average Loss: {avg_loss:.3f}")
            # print(f"Average Loss: {avg_loss:.4f}")
            
            # Save checkpoints
            if (epoch + 1) % 5 == 0:
                self.save_learned_embeddings(tokens_to_train, f'embeddings_epoch_{epoch+1}.pt')

    def train_tokens2(self, dataloader, train_config = None):
        
        # Create optimizer only for special token embeddings
        # self.optimizer = torch.optim.AdamW([
        #     {'params': self.token_embeddings.values(), 
        #      'lr': train_config['learning_rate'],
        #      'weight_decay': 0.01}
        # ])
        self.optimizer = torch.optim.AdamW([
            {'params': [self.token_embedding.weight[id] for id in self.token_ids.values()], 
             'lr': train_config['learning_rate'],
             'weight_decay': 0.01}
        ])

        num_training_steps  = len(dataloader) * train_config['num_epochs']
        num_warmup_steps    = num_training_steps // 10
        # Optionally, enable a learning rate scheduler
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=num_warmup_steps,
        #     num_training_steps=num_training_steps
        # )
        
        best_loss           = float('inf')
        patience            = 200
        patience_counter    = 0
        running_avg_loss    = []


        for epoch in range(train_config["num_epochs"]):
            self.model.train() 
            total_loss = 0
            num_batches = 0
            # for images, prompts, tokens in tqdm(dataloader, desc=f"Epoch {epoch+1}/{train_config['num_epochs']}"):
            for images, prompts, tokens in dataloader:
                # print(prompts, tokens)
                images = (2 * images - 1).to(self.device) # Normalize images to [-1, 1]
                loss = self.train_step2(images, prompts)
                total_loss += loss
                num_batches += 1
            avg_loss = total_loss / num_batches
            running_avg_loss.append(avg_loss)
            smoothed_loss = sum(running_avg_loss[-5:]) / min(len(running_avg_loss), 5)
            print(f"Epoch {epoch+1}/{train_config['num_epochs']}, Average Loss: {avg_loss:.4f}, Smooth Loss {smoothed_loss:.4f}")
            # scheduler.step(smoothed_loss)
            
            if smoothed_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'embeddings': self.token_embeddings,
                    'optimizer_state': self.optimizer.state_dict(),
                }, 'best_embeddings.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    break
            self.check_gradients()
        checkpoint = torch.load('best_embeddings.pt')
        self.token_embeddings = checkpoint['embeddings']

    def train_step2(self, images, prompts):
        self.optimizer.zero_grad()        
        with torch.no_grad():
            latents = self.model.encode_first_stage(images)
            latents = self.model.get_first_stage_encoding(latents)
        latents *=  0.18215
        
        # Create timesteps
        batch_size = images.shape[0]
        timesteps = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise to latents
        noise = torch.randn_like(latents).to(self.device)
        noisy_latents = self.model.q_sample(latents, timesteps, noise).to(self.device)
        
        # Prepare text embeddings with special token
        with torch.set_grad_enabled(True):  # Enable gradients for text embeddings
            text_embeddings = self.get_text_embeddings(prompts[0]).to(self.device)

            # text_embeddings = self.model.get_learned_conditioning(prompts[0])
        #     uncond_embeddings = self.model.get_learned_conditioning([""] * images.shape[0])
        #     text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(self.device)
        # noisy_latents = torch.cat([noisy_latents] * 2)
        # timesteps = torch.cat([timesteps] * 2)

        # Predict noise
        noise_pred = self.model.apply_model(noisy_latents, timesteps, text_embeddings)

    
        main_loss = self.criterion(noise_pred, noise)
        # Add regularization for token embeddings
        reg_loss = 0.0
        for emb in self.token_embeddings.values():
            reg_loss += 0.01 * torch.norm(emb, p=2)
        total_loss = main_loss + reg_loss
        total_loss.backward()

        # Backward pass with gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [emb for emb in self.token_embeddings.values()], 
            max_norm=1.0
        )

        self.optimizer.step()

        # Normalize token embeddings after optimization step
        with torch.no_grad():
            for token in self.token_embeddings:
                self.token_embeddings[token].data = torch.nn.functional.normalize(
                    self.token_embeddings[token].data, 
                    dim=-1
                )
        
        return main_loss.item()

    def get_text_embeddings(self, prompts):
        # Get base token ids from tokenizer
        logger.debug(f'Current Get Text Embedding: {prompts[0]}')
        
        # Make sure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Tokenize with proper padding
        tokens = self.model.cond_stage_model.tokenizer(
            prompts,
            max_length=self.model.cond_stage_model.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get base embeddings
        input_ids = tokens["input_ids"]
        base_embeddings = self.model.cond_stage_model.transformer.text_model.embeddings.token_embedding(input_ids)
        
        # Create embedding mask to track which tokens we've replaced
        embedding_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Replace special token embeddings
        for token in self.special_tokens:
            # Get token ID from tokenizer
            token_id = self.model.cond_stage_model.tokenizer.convert_tokens_to_ids(token)
            
            # Find positions of this token
            token_positions = (input_ids == token_id).nonzero(as_tuple=True)
            
            if token_positions[0].numel() > 0:
                # Replace embeddings at those positions
                batch_idx, seq_positions = token_positions
                base_embeddings[batch_idx, seq_positions] = self.token_embeddings[token]
                embedding_mask[batch_idx, seq_positions] = False
                
                logger.debug(f"Replaced embeddings for token '{token}' at positions {seq_positions.tolist()}")
        
        # Add position embeddings
        position_ids = tokens["attention_mask"].long() * torch.arange(self.model.cond_stage_model.max_length).to(self.device)
        position_embeddings = self.model.cond_stage_model.transformer.text_model.embeddings.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = base_embeddings + position_embeddings
        
        # Create attention mask
        attention_mask = tokens["attention_mask"]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.expand(-1, -1, attention_mask.size(1), -1)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Pass through transformer
        encoder_outputs = self.model.cond_stage_model.transformer.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        
        # Apply final layer norm
        text_embeddings = self.model.cond_stage_model.transformer.text_model.final_layer_norm(
            encoder_outputs.last_hidden_state
        )
        
        # Normalize embeddings
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings
    
    def get_text_embeddings2(self, text):
        """Get text embeddings for the given text"""
        tokenizer = self.model.cond_stage_model.tokenizer
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Create base embeddings
        base_embeddings = torch.zeros((1, len(tokens), self.embedding_dim)).to(self.device)
        
        # Fill in embeddings
        for batch_idx in range(1):
            for seq_idx, token in enumerate(tokens):
                if token in self.token_embeddings:
                    base_embeddings[batch_idx, seq_idx] = self.token_embeddings[token]
                else:
                    base_embeddings[batch_idx, seq_idx] = self.token_embedding.weight.data[token_ids[seq_idx]]
                    
        return base_embeddings

    def check_gradients(self):
        for token, emb in self.token_embeddings.items():
            if emb.grad is not None:
                grad_mean = emb.grad.mean().item()
                grad_std = emb.grad.std().item()
                logger.debug(f"Gradient for {token}: Mean={grad_mean:.5e}, Std={grad_std:.5e}")
                
                # Optional checks for zero or abnormal gradients
                if grad_mean == 0 or grad_std == 0:
                    logger.debug(f"Warning: Gradient for {token} is zero.")
                if grad_mean > 1e3 or grad_std > 1e3:
                    logger.debug(f"Warning: High gradient for {token}, consider clipping.")
            else:
                logger.debug(f"Warning: Gradient for {token} is None (not computed or detached).")

    def load_learned_embeddings2(self, checkpoint_path):
        """Load trained embeddings from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if 'embeddings' in checkpoint:
                # New format with dictionary
                self.token_embeddings = checkpoint['embeddings']
            else:
                # Old format with direct embeddings
                self.token_embeddings = checkpoint
        
        logger.info(f"Loaded embeddings for tokens: {list(self.token_embeddings.keys())}")

    def generate_images(self, prompt, idx, num_iter=1, config={}, save_num = 25):
        """Generate multiple images using the trained model"""
        print(f"Generating images using prompt: {prompt}")
        os.makedirs(config["output_folder"], exist_ok=True)
        sampler = DPMSolverSampler(self.model)

        batch_size  = config["inf_batch_size"]
        data        = [batch_size * [prompt]]
        shape       = [config["latent_channels"], 
                    config["height"] // config["downsample_factor"], 
                    config["width"] // config["downsample_factor"]]
        start_code  = None

        token_in_prompt = [token for token in self.special_tokens if token in prompt]
        print(f'token_in_prompt: {token_in_prompt}')
        
        with torch.no_grad():
            with self.precision_scope(self.device):
                base_count = 0
                for n in range(num_iter):
                    for prompts in data:
                        uc = None
                        if config["guidance_scale"] != 1.0:
                            uc = self.model.get_learned_conditioning(batch_size * [""])
                        if token_in_prompt:
                            c = self.get_text_embeddings(prompts)
                        else:
                            # c = self.model.get_learned_conditioning(prompts)
                            c = self.get_text_embeddings(prompts)
                        samples_ddim, _ = sampler.sample(S=config["ddim_steps"],
                                                        conditioning=c,
                                                        batch_size=batch_size,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=config["guidance_scale"],
                                                        unconditional_conditioning=uc,
                                                        eta=config["ddim_eta"],
                                                        x_T=start_code)

                        
                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            # img = put_watermark(img, wm_encoder)
                            save_path = os.path.join(config["output_folder"], f"prompt_{idx}_sample_{base_count+1}.png")
                            img.save(save_path)
                            base_count += 1
                            if base_count == save_num:
                                return
                            print(f"Saved image to {save_path}")


def main():
    config = {
        "config_path": "./configs/stable-diffusion/v1-inference.yaml",
        "model_path": "sd-v1-4.ckpt",
        "do_train" :1,
        "do_sample" :1,
        "new1_img_dir": "../hw2_data/textual_inversion/0",
        "new2_img_dir": "../hw2_data/textual_inversion/1",
        "num_epochs": 400,
        "trian_imgz" : 512,
        "train_batch_size" :1,
        "inf_batch_size": 1,
        "learning_rate": 1e-4,
        "output_folder": "./output",
        "latent_channels" : 4,
        "downsample_factor" : 8,
        "height" : 512,
        "width": 512,
        "guidance_scale" : 7.5, #unconditional guidance scale
        "precision" : "autocast",
        "ddim_steps" : 50,
        "ddim_eta" : 0.0
    }

    device          = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    tokens_to_train = ['<new1>', '<new2>']
    ti              = TextualInversion(config["config_path"], config["model_path"], tokens_to_train,
                        device = device, precision_scope = config["precision"] )

    dataset1 = CustomDataset(config["new1_img_dir"], token=tokens_to_train[0], imgz = config["trian_imgz"])
    dataset2 = CustomDataset(config["new2_img_dir"], token=tokens_to_train[1], imgz = config["trian_imgz"])
    combined_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    dataloader = DataLoader(combined_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers = 4)

    if config["do_train"]:
        # ti.train_tokens(
        #     dataloader=dataloader,
        #     train_config = config
        # )
        ti.train_tokens2(
            dataloader=dataloader,
            train_config = config
        )

    test_prompts = [
        "A photo of <new1>.",
        "A <new1> shepherd posing proudly on a hilltop with Mount Fuji in the background.",
        "A <new1> perched on a park bench with the Colosseum looming behind.",
        "The streets of Paris in the style of <new2>.",
        "Manhattan skyline in the style of <new2>."
    ]
    
    start_inf_time = time.time()
    if config["do_sample"]:
        # ti.load_learned_embeddings("embeddings_epoch_20.pt")
        ti.load_learned_embeddings2("best_embeddings.pt")
        for idx ,prompt in enumerate(test_prompts):
            ti.generate_images(
                prompt          =prompt,
                idx             =idx,
                num_iter        =30,
                config          =config,
                save_num        =10
            )
            # ti.generate_images_selfModel(
            #     prompt          =prompt,
            #     idx             =idx,
            #     num_iter        =30,
            #     config          =config,
            #     save_num        =10
            # )
    print(f'Using {time.time() - start_inf_time:.3f} sec to generate images')

if __name__ == "__main__":
    main()
