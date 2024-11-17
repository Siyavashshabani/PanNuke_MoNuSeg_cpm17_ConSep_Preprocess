import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage.morphology import erosion, disk

import numpy as np 
import os
from PIL import Image
import numpy as np
from scipy.io import loadmat
import torch 
from skimage.morphology import erosion, disk
import numpy as np
from skimage import measure


def remove_small_border_cells(binary_mask, size_threshold):
    """
    Removes cells that are both on the borders of the image and smaller than a given size threshold.
    
    Parameters:
        binary_mask (numpy.ndarray): The binary image mask where cells are marked.
        size_threshold (int): The minimum size; cells smaller than this and on the border are removed.
    
    Returns:
        numpy.ndarray: The cleaned binary mask.
    """
    # Label connected components
    labels = measure.label(binary_mask)
    properties = measure.regionprops(labels)

    # Create a mask where cells that are both small and border-touching are removed
    clean_mask = np.copy(labels)

    # Check each region to see if it meets both criteria
    for prop in properties:
        # Check if the region touches the border
        min_row, min_col, max_row, max_col = prop.bbox
        touches_border = min_row == 0 or min_col == 0 or max_row == labels.shape[0] or max_col == labels.shape[1]

        # Check if the region is smaller than the threshold
        is_small = prop.area < size_threshold

        # Remove the cell if it touches the border and is small
        if touches_border and is_small:
            clean_mask[labels == prop.label] = 0

    # Convert cleaned labels back to binary
    return clean_mask  #clean_mask > 0


def instance_map_to_channels(instance_map):
    """
    Convert an instance map with unique identifiers for each cell into a multi-channel binary mask.
    
    Parameters:
        instance_map (numpy.ndarray): A 2D array where each unique value (except for zero if used for background) 
                                      represents a different cell.
    
    Returns:
        numpy.ndarray: A 3D array where the first dimension is the number of unique cells and each channel 
                       is a binary mask for one cell.
    """
    # Extract unique cell identifiers, ignoring zero if it's used for background
    unique_cells = np.unique(instance_map)
    unique_cells = unique_cells[unique_cells != 0]  # Adjust this line if 0 should not be ignored

    # Prepare output array with shape [num_cells, width, height]
    num_cells = len(unique_cells)
    channels = np.zeros((num_cells, instance_map.shape[0], instance_map.shape[1]), dtype=np.uint8)

    # Create a binary mask for each cell
    for index, cell in enumerate(unique_cells):
        channels[index] = (instance_map == cell).astype(np.uint8)

    return channels


def load_and_preprocess_image(img_path):
    img = Image.open(img_path)
    img_array = np.array(img) / 255.0  # Normalize image
    if img_array.shape[2] == 4:  # Handle RGBA images
        img_array = img_array[:, :, :3]
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
    while y + window_size <= img.shape[0]:  # Ensure within vertical bounds
        x = 0
        while x + window_size <= img.shape[1]:  # Ensure within horizontal bounds
            patch = img[y:y + window_size, x:x + window_size]
            if patch_type == "mask" and remove_cells_borders:
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
            mask_filename = filename.replace('.png', '.mat')
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

    parser = argparse.ArgumentParser(description="Process a dataset for image and mask patch extraction.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., ConSep)")
    parser.add_argument("--subset", type=str, required=True, choices=["train", "test"], help="Subset to process (train or test)")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for the dataset (e.g., /home/user/projects/MoNuSeg)")

    args = parser.parse_args()

    dataset = args.dataset
    subset = args.subset
    base_dir = args.base_dir

    image_directory = os.path.join(base_dir, dataset, subset, "Images")
    mask_directory = os.path.join(base_dir, dataset, subset, "Labels")
    output_folder = os.path.join(base_dir, dataset, "preprocessed", subset, "images")
    output_mask_folder = os.path.join(base_dir, dataset, "preprocessed", subset, "labels")

    main(image_directory, mask_directory, output_folder, output_mask_folder, dataset, erosion=False, remove_cells_borders=True)


## How to run the code :python process_dataset.py --dataset ConSep --subset train --base_dir ./

## How to run the code :python process_dataset.py --dataset ConSep --subset test --base_dir ./

