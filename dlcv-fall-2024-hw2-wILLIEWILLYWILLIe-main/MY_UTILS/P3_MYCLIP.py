import os
import torch
import clip
import json
from PIL import Image
from tqdm import tqdm

image_dir   = "./hw2_data/clip_zeroshot/val"
label_json  = "./hw2_data/clip_zeroshot/id2label.json"

with open(label_json, 'r') as f:
    id_to_label = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text_prompts = [f"A photo of a {label}." for label in id_to_label.values()]
text_tokens = clip.tokenize(text_prompts).to(device)

def get_class_from_filename(filename):
    return int(filename.split('_')[0])

# Evaluation loop
correct_predictions = 0
total_predictions = 0
successful_cases = []
failed_cases = []

for image_file in tqdm(os.listdir(image_dir)):
    if image_file.endswith(".png"):
        image_path = os.path.join(image_dir, image_file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        true_class_id = get_class_from_filename(image_file)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)

            logits_per_image, logits_per_text = model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        predicted_class_id = probs.argmax()

        total_predictions += 1
        if predicted_class_id == true_class_id:
            correct_predictions += 1
            successful_cases.append((image_file, id_to_label[str(predicted_class_id)]))
        else:
            failed_cases.append((image_file, id_to_label[str(predicted_class_id)], id_to_label[str(true_class_id)]))

accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%, total_pred {total_predictions}")

# Report successful and failed cases
print("\nSuccessful cases:")
for success in successful_cases[:5]:
    print(success)

print("\nFailed cases:")
for failure in failed_cases[:5]:
    print(f"Predicted: {failure[1]}, Actual: {failure[2]} - Image: {failure[0]}")