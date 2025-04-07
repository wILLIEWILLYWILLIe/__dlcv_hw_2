import torch
import numpy as np
import os
import random
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image 
from torchvision import transforms

seed_num = 100
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)

class DigitsDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform = transforms.ToTensor()):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        # self.__init_check()
        
    def __init_check(self):
        print("Checking all image paths and labels:")
        for idx in range(len(self.data)):
            image_name = self.data.iloc[idx]['image_name']
            label = self.data.iloc[idx]['label']
            image_path = os.path.join(self.image_dir, image_name)
            
            print(f"Image {idx}: Path: {image_path}, Label: {label}")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image path {image_path} does not exist.")
        print("Image path and label check completed.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['image_name']
        label = self.data.iloc[idx]['label']
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image_path, image, label

class DigitsDataset_list(Dataset):
    def __init__(self, csv_paths, dir_paths, transform=None):
        """
        Args:
            csv_paths (list): List of CSV file paths (can be 1 or more).
            dir_paths (list): List of corresponding directory paths for images (can be 1 or more).
            transform (callable): Transform to be applied to the images (default: ToTensor()).
        """
        # Ensure the number of CSV paths matches the number of directory paths
        assert len(csv_paths) == len(dir_paths), "Number of CSV paths must match number of directory paths."
        
        self.data_list = []  
        self.dir_paths = dir_paths
        if transform is None:  
            self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  
            transforms.ToTensor(),
            ])
        else:self.transform = transform

        
        # Load data from each CSV
        for csv_path in csv_paths:
            data = pd.read_csv(csv_path)
            self.data_list.append(data)
        
        self.data = pd.concat(self.data_list, ignore_index=True)
        
        # self.__init_check()

    def __init_check(self):
        print("Checking all image paths and labels:")
        
        for dataset_idx, dataset in enumerate(self.data_list):
            dataset_dir = self.dir_paths[dataset_idx]
            
            for idx in range(len(dataset)):
                image_name = dataset.iloc[idx]['image_name']
                label = dataset.iloc[idx]['label']
                image_path = os.path.join(dataset_dir, image_name)
                
                print(f"Dataset {dataset_idx}, Image {idx}: Path: {image_path}, Label: {label}")
                if not os.path.exists(image_path):
                    print(f"Warning: Image path {image_path} does not exist.")
        
        print("Image path and label check completed.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        running_len = 0
        
        for dataset_idx, dataset in enumerate(self.data_list):
            if idx < running_len + len(dataset):
                # Found the right dataset, adjust the index and get the corresponding data
                actual_idx = idx - running_len
                image_name = dataset.iloc[actual_idx]['image_name']
                label = dataset.iloc[actual_idx]['label']
                image_path = os.path.join(self.dir_paths[dataset_idx], image_name)
                
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
                return image_path, image, label, dataset_idx
            running_len += len(dataset)

        raise IndexError(f"Index {idx} out of range for DigitsDataset")

def add_noise(image_tensor, noise_factor=0.1):
    noise = torch.randn_like(image_tensor) * noise_factor  
    noisy_image = image_tensor + noise  
    noisy_image = torch.clamp(noisy_image, 0, 1) 
    return noisy_image

def generate_and_save_images(loader, output_folder, dataset_name, num_images=50):
    os.makedirs(output_folder, exist_ok=True)
    for target in range(10):  # Digits from 0 to 9
        count = 0
        for i, (image_path, image, label) in enumerate(loader):
            if label.item() == target :
                if count < num_images:

                    noisy_image = add_noise(image, noise_factor=0.1)
                    filename = f"{target}_{count+1:03}.png"
                    file_path = os.path.join(output_folder, filename)
                    # save_image(noisy_image, file_path)
                    
                    count += 1
                    # print(f"Generated {dataset_name}: {filename} at {file_path} from {image_path}")
                else:
                    break

if __name__ == "__main__" : 
    MNISTM_OUTPUT_DIR   = "./output_folder_example/mnistm"
    SVHN_OUTPUT_DIR     = "./output_folder_example/svhn"

    MNISTM_DATA_DIR = "./hw2_data/digits/mnistm/data" 
    MNISTM_CSV      = "./hw2_data/digits/mnistm/train.csv"
    SVHN_DATA_DIR   = "./hw2_data/digits/svhn/data"
    SVHN_CSV        = "./hw2_data/digits/svhn/train.csv"

    # mnistm_dataset  = DigitsDataset(MNISTM_CSV, MNISTM_DATA_DIR)
    # svhn_dataset    = DigitsDataset(SVHN_CSV, SVHN_DATA_DIR)

    # mnistm_loader   = DataLoader(mnistm_dataset, batch_size=1, shuffle=False)
    # svhn_loader     = DataLoader(svhn_dataset, batch_size=1, shuffle=False)

    # generate_and_save_images(mnistm_loader, MNISTM_OUTPUT_DIR, "MNIST-M")
    # generate_and_save_images(svhn_loader, SVHN_OUTPUT_DIR, "SVHN")
