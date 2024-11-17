import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage.morphology import erosion, disk
from skimage import measure

def instance_map_to_channels(instances):
    """
    Converts an instance map to channel-wise binary masks.
    """
    unique_instances = np.unique(instances)
    channel_maps = np.zeros((len(unique_instances), *instances.shape), dtype=np.uint8)
    for idx, instance_id in enumerate(unique_instances):
        if instance_id == 0:  # Ignore background
            continue
        channel_maps[idx] = (instances == instance_id).astype(np.uint8)
    return channel_maps

def remove_small_border_cells(patch, size_threshold):
    """
    Removes small cells touching the patch borders.
    """
    labeled_patch = measure.label(patch, connectivity=1)
    for region in measure.regionprops(labeled_patch):
        if region.area < size_threshold:
            coords = region.coords
            if np.any(coords[:, 0] == 0) or np.any(coords[:, 1] == 0) or \
               np.any(coords[:, 0] == patch.shape[0] - 1) or np.any(coords[:, 1] == patch.shape[1] - 1):
                patch[coords[:, 0], coords[:, 1]] = 0
    return patch

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
        img_patch_filename = f"{dataset}_{base_filename}_image_{global_patch_index}.npy"
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
                patch[patch >= 0.5] = 1
                patch[patch < 0.5] = 0
                if remove_cells_borders:
                    patch = remove_small_border_cells(patch, size_threshold=50)
            patches.append(patch)
            x += stride
        y += stride
    return patches

def main(image_directory, mask_directory, output_folder, output_mask_folder, dataset, erosion_flag, remove_cells_borders):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    i = 0
    for filename in os.listdir(image_directory):
        if filename.endswith(".tif"):
            img_path = os.path.join(image_directory, filename)
            mask_filename = filename.replace(".tif", "_mask.png")
            mask_path = os.path.join(mask_directory, mask_filename)

            img_array = load_and_preprocess_image(img_path)

            if erosion_flag:
                instance_argmax_map = load_and_process_mask(mask_path)
            else:
                instance_argmax_map = np.array(Image.open(mask_path))

            mask_patches = extract_patches(instance_argmax_map, remove_cells_borders, window_size=256, stride=128, patch_type="mask")
            img_patches = extract_patches(img_array, remove_cells_borders, window_size=256, stride=128, patch_type="image")
            save_patches(img_patches, mask_patches, output_folder, output_mask_folder, dataset, filename[:-4])
            i += 1
            print(f"Image number {i} processed!")

# Main script
if __name__ == "__main__":
    print("Start processing the train dataset.")
    subset = "train"
    image_directory = f"./MoNuSeg/{subset}/images"
    mask_directory = f"./MoNuSeg/{subset}/masks"
    output_folder = f"./MoNuSeg/preprocessed/fold0/images"
    output_mask_folder = f"./MoNuSeg/preprocessed/fold0/labels"
    dataset = "MoNuSeg"
    main(image_directory, mask_directory, output_folder, output_mask_folder, dataset, erosion_flag=False, remove_cells_borders=True)

    print("Start processing the test dataset.")
    subset = "test"
    image_directory = f"./MoNuSeg/{subset}/images"
    mask_directory = f"./MoNuSeg/{subset}/masks"
    output_folder = f"./MoNuSeg/preprocessed/fold1/images"
    output_mask_folder = f"./MoNuSeg/preprocessed/fold1/labels"
    dataset = "MoNuSeg"
    main(image_directory, mask_directory, output_folder, output_mask_folder, dataset, erosion_flag=False, remove_cells_borders=True)
