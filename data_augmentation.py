import os
import json
from PIL import Image, ImageOps
from tqdm import tqdm

def process_images_with_descriptions(input_dir, json_path, output_dir, output_json_path):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load descriptions from input JSON
    with open(json_path, 'r') as f:
        descriptions = json.load(f)

    new_descriptions = {}

    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            if filename not in descriptions:
                print(f"Warning: {filename} not found in JSON, skipping.")
                continue

            # Open image
            img = Image.open(img_path)

            # Save original image to new directory
            original_out_name = filename
            original_out_path = os.path.join(output_dir, original_out_name)
            img.save(original_out_path)
            new_descriptions[original_out_name] = descriptions[filename]

            # Create mirrored image if there is more than one
            mirrored_img = ImageOps.mirror(img)

            # Save mirrored image with new name
            mirrored_out_name = os.path.splitext(filename)[0] + '_mirrored' + os.path.splitext(filename)[1]
            mirrored_out_path = os.path.join(output_dir, mirrored_out_name)
            mirrored_img.save(mirrored_out_path)

            # Add description for mirrored image (same as original)
            new_descriptions[mirrored_out_name] = descriptions[filename]

    # Save new JSON with all image descriptions
    with open(output_json_path, 'w') as f:
        json.dump(new_descriptions, f, indent=4)

    print(f"Processing complete. Output saved to: {output_dir} and {output_json_path}")

if  __name__=="__main__":
    process_images_with_descriptions(
        input_dir='iclevr',
        json_path='train.json',
        output_dir='augmented_iclv',
        output_json_path='augmented_train.json'
    )
