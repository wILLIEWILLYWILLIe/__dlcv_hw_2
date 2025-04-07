import os
import time
import json
import numpy as np
import sys
import torch
from PIL import Image
from tqdm import tqdm

import importlib.util
import subprocess

module_path = os.path.join(os.path.dirname(__file__), "stable-diffusion", "P3_try2.py")
spec = importlib.util.spec_from_file_location("P3_try2", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules["P3_try2"] = module
spec.loader.exec_module(module)

# Access classes and functions
TextualInversion    = module.TextualInversion
ti_config           = module.ti_config

command = [
    "python3", "evaluation/grade_hw2_3.py",
    "--json_path", "./hw2_data/textual_inversion/input.json",
    "--input_dir", "./hw2_data/textual_inversion",
    "--output_dir", "./output_folder"
]
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
'''
def main(json_file, output_folder, model_weights, ckpt_pth1 = None, ckpt_pth2 = None):
    # Load config and model
    EVAL_TEST = 0
    config                      = ti_config.copy()
    config["config_path"]       =  "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    config["model_path"]        =  "./stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt"
    config["model_path"]        =  model_weights
    config["output_folder"]     =  output_folder
    if ckpt_pth1 is None:
        # ckpt_pth1                   = "./stable-diffusion/epoch580_ckpt.pth"
        ckpt_pth1                   = "./stable-diffusion/epoch800_1_ckpt.pth"
    if ckpt_pth2 is None:
        ckpt_pth2               = "./stable-diffusion/epoch600new_ckpt.pth"
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    ti      = TextualInversion( config["config_path"], 
                                config["model_path"], 
                                config["tokens_to_train"],
                                device=device, 
                                train_config=config["train_config"], 
                                inf_config = config["sample_config"]
    )
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if not EVAL_TEST:
        start_inf_time = time.time()
        ti.load_checkpoint(ckpt_pth1, token_ids_to_update=[49408])
        ti.load_checkpoint(ckpt_pth2, token_ids_to_update=[49409])
        for src_idx, sample in data.items():
            prompt_list = sample['prompt']
            for prompt_idx, prompt in enumerate(prompt_list):
                # if src_idx == '0' and prompt_idx == 1:
                # if src_idx == '1' :
                if True:
                    src_prompt_outputdir = os.path.join(output_folder, f'{src_idx}', f'{prompt_idx}')
                    ti.inference(prompt_template = prompt, 
                                        num_iter=30, 
                                        save_num=25, 
                                        save_path=f'source{src_idx}_prompt{prompt_idx}', 
                                        save_dir=src_prompt_outputdir
                                        )
        print(f"Using {time.time() - start_inf_time:.3f} sec to generate images")

    else :
        start_inf_time = time.time()
        ckpt1_epoch = [0,5,10,50,75,100]
        
        # ckpt2_epoch = [0,5,10,50,75,100] + list(range(100, 1000, 100)) + list(range(400, 901, 20))
        # ckpt2_epoch = list(range(100, 1000, 100)) + list(range(400, 901, 20))
        ckpt2_epoch = list(range(500, 1000, 100)) + list(range(600, 901, 20))
        ckpt2_epoch = sorted(set(ckpt2_epoch))
        
        # for epoch in ckpt1_epoch:
        #     ckpt_pth1  =  f"./stable-diffusion/ckpt_folder_new/epoch{epoch}_1_ckpt.pth"
        # for epoch in ckpt2_epoch:
        #     ckpt_pth2  = f"./stable-diffusion/ckpt_folder_new/epoch{epoch}_2_ckpt.pth"
        # eta_range = [0,0.1]
        # for _eta in eta_range:
        #     print(f"_eta --> {_eta}")
        # ddim_step_range = [90,100]
        # for _ddimstep in ddim_step_range:
        #     print(f"_ddimstep --> {_ddimstep}")
        scale_range = [5,5.5,6,6.5,7,7.5,8,9]
        for _scale in scale_range:
            print(f"_scale --> {_scale}")

            ti.load_checkpoint(ckpt_pth1, token_ids_to_update=[49408])
            ti.load_checkpoint(ckpt_pth2, token_ids_to_update=[49409])
            start_inf_time = time.time()
            for src_idx, sample in data.items():
                prompt_list = sample['prompt']
                for prompt_idx, prompt in enumerate(prompt_list):
                    if src_idx == '0' and prompt_idx == 1:
                    # if src_idx == '1' :
                    # if True:
                        src_prompt_outputdir = os.path.join(output_folder, f'{src_idx}', f'{prompt_idx}')
                        ti.inference(prompt_template = prompt, 
                                            num_iter=30, 
                                            save_num=25, 
                                            save_path=f'source{src_idx}_prompt{prompt_idx}', 
                                            save_dir=src_prompt_outputdir,
                                            scale = _scale,
                                            eta = 0.1,
                                            ddimstep = 70
                                            )
            print(f"Using {time.time() - start_inf_time:.3f} sec to generate iamges")
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Command output:", result.stdout)

if __name__ == "__main__":
    import sys
    json_file       = sys.argv[1]
    output_folder   = sys.argv[2]
    model_weights   = sys.argv[3]
    main(json_file, output_folder, model_weights)