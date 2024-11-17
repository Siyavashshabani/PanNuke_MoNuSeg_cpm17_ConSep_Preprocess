import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage.morphology import erosion, disk

def load_and_preprocess_image(img_path):
    img = Image.open(img_path)
    img_array = np.array(img) / 255.0  # Normalize image
    return img_array

def load_and_process_mask(mask_path):
    instances = loadmat(mask_path)['inst_map']  # Ensure the key matches your .mat file
    instance_maps = instance_map_to_channels(instances)
    for i in range(instance_maps.shape[0]):
        instance_maps[i] = erosion(instance_maps[i], disk(1))
    instance_argmax_map = np.argmax(instance_maps, axis=0)
    return instance_argmax_map

def save_patches(img_patches, mask_patches, output_folder, output_mask_folder, dataset, base_filename):
    global_patch_index = 0

    for img_patch, mask_patch in zip(img_patches, mask_patches):
        img_patch_filename = f"{dataset}_{base_filename}_{global_patch_index}.npy"
        mask_patch_filename = img_patch_filename.replace("image", "mask")
        np.save(os.path.join(output_folder, img_patch_filename), img_patch)
        np.save(os.path.join(output_mask_folder, mask_patch_filename), mask_patch)
        global_patch_index += 1

def extract_patches(img, remove_cells_borders, window_size, stride, patch_type):
    """Extract patches from a given image."""
    patches = []
    y = 0
    while y + window_size <= img.shape[0]:  # ensure within vertical bounds
        x = 0
        while x + window_size <= img.shape[1]:  # ensure within horizontal bounds
            patch = img[y:y + window_size, x:x + window_size]
            if patch_type == "mask":
                if remove_cells_borders:
                    patch = remove_small_border_cells(patch, size_threshold=50)
            patches.append(patch)
            x += stride
        y += stride
    return patches

def main(image_directory, mask_directory, output_folder, output_mask_folder, dataset, erosion, remove_cells_borders):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    for filename in os.listdir(image_directory):
        if filename.endswith(".png"):
            img_path = os.path.join(image_directory, filename)
            mask_filename = filename.replace(".png", ".mat")
            mask_path = os.path.join(mask_directory, mask_filename)

            img_array = load_and_preprocess_image(img_path)

            if erosion:
                instance_argmax_map = load_and_process_mask(mask_path)
            else:
                instance_argmax_map = loadmat(mask_path)['inst_map']

            mask_patches = extract_patches(instance_argmax_map, remove_cells_borders, window_size=256, stride=64, patch_type="mask")
            img_patches = extract_patches(img_array, remove_cells_borders, window_size=256, stride=64, patch_type="image")
            save_patches(img_patches, mask_patches, output_folder, output_mask_folder, dataset, filename[:-4])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process dataset for patch extraction.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., CPM17)")
    parser.add_argument("--subset", type=str, required=True, choices=["train", "test"], help="Subset to process (train or test)")
    args = parser.parse_args()

    dataset = args.dataset
    subset = args.subset

    image_directory = f"./{dataset}/{subset}/Images"
    mask_directory = f"./{dataset}/{subset}/Labels"
    output_folder = f"./{dataset}/preprocessed/{subset}/images"
    output_mask_folder = f"./{dataset}/preprocessed/{subset}/labels"

    main(image_directory, mask_directory, output_folder, output_mask_folder, dataset, erosion=False, remove_cells_borders=True)


#How to run: python process_dataset.py --dataset CPM17 --subset train
#How to run: python process_dataset.py --dataset CPM17 --subset test
