import numpy as np
import os

def save_images_and_masks(images, masks, types, images_dir, masks_dir, tissues_dir, Neoplastic_dir, inflams_dir, Connective_dir, Dead_dir, Epithelial_dir, fold):
    """
    Save images and masks to separate .npy files.

    Args:
        images (numpy.ndarray): The images array with shape (n, 256, 256, 3).
        masks (numpy.ndarray): The masks array with shape (n, 256, 256, 1).
        types (numpy.ndarray): The types array indicating the type of each image.
        images_dir (str): Directory to save images.
        masks_dir (str): Directory to save masks.
        tissues_dir (str): Directory to save tissues.
        Neoplastic_dir (str): Directory to save Neoplastic masks.
        inflams_dir (str): Directory to save inflammatory masks.
        Connective_dir (str): Directory to save Connective masks.
        Dead_dir (str): Directory to save Dead masks.
        Epithelial_dir (str): Directory to save Epithelial masks.
        fold (int): Fold number to include in the filename.

    Returns:
        None
    """
    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(tissues_dir, exist_ok=True)
    os.makedirs(Neoplastic_dir, exist_ok=True)
    os.makedirs(inflams_dir, exist_ok=True)
    os.makedirs(Connective_dir, exist_ok=True)
    os.makedirs(Dead_dir, exist_ok=True)
    os.makedirs(Epithelial_dir, exist_ok=True)

    # Iterate over each index and save the corresponding image and mask
    for i in range(masks.shape[0]):
        # Define file paths
        image_path = os.path.join(images_dir, f'PanNuke_fold{fold}_{types[i]}_image_{i}.npy')
        mask_path = os.path.join(masks_dir, f'PanNuke_fold{fold}_{types[i]}_mask_{i}.npy')
        tissue_path = os.path.join(tissues_dir, f'PanNuke_fold{fold}_{types[i]}_tissueType_{i}.npy')

        Neoplastic_path = os.path.join(Neoplastic_dir, f'PanNuke_fold{fold}_{types[i]}_Neoplastic_{i}.npy')
        inflam_path = os.path.join(inflams_dir, f'PanNuke_fold{fold}_{types[i]}_Inflam_{i}.npy')
        Connective_path = os.path.join(Connective_dir, f'PanNuke_fold{fold}_{types[i]}_Connective_{i}.npy')
        Dead_path = os.path.join(Dead_dir, f'PanNuke_fold{fold}_{types[i]}_Dead_{i}.npy')
        Epithelial_path = os.path.join(Epithelial_dir, f'PanNuke_fold{fold}_{types[i]}_Epithelial_{i}.npy')

        # Create the instances
        mask = np.max(masks[i, :, :, :5], axis=-1, keepdims=True)
        tissue = masks[i, :, :, :]
        Neoplastic = masks[i, :, :, 0]
        inflammatory = masks[i, :, :, 1]
        Connective = masks[i, :, :, 2]
        Dead = masks[i, :, :, 3]
        Epithelial = masks[i, :, :, 4]

        # Save each image and mask
        np.save(image_path, images[i] / 255.0)
        np.save(mask_path, mask)
        np.save(tissue_path, tissue)
        np.save(Neoplastic_path, Neoplastic)
        np.save(inflam_path, inflammatory)
        np.save(Connective_path, Connective)
        np.save(Dead_path, Dead)
        np.save(Epithelial_path, Epithelial)

        # Optional: Print out the file paths of saved files
        # print(f'Saved image to {image_path}')
        # print(f'Saved mask to {mask_path}')

# Main script
if __name__ == "__main__":
    for fold in [1, 2, 3]:
        images_dir = f"./PanNuke/preped/fold{fold}/images"
        masks_dir = f"./PanNuke/preped/fold{fold}/masks"
        tissues_dir = f"./PanNuke/preped/fold{fold}/tissues"
        Neoplastic_dir = f"./PanNuke/preped/fold{fold}/Neoplastic"
        inflams_dir = f"./PanNuke/preped/fold{fold}/inflams"
        Connective_dir = f"./PanNuke/preped/fold{fold}/Connective"
        Dead_dir = f"./PanNuke/preped/fold{fold}/Dead"
        Epithelial_dir = f"./PanNuke/preped/fold{fold}/Epithelial"

        # Read the data
        masks = np.load(f"./PanNuke/raw_data/Fold {fold}/masks/fold{fold}/masks.npy")
        types = np.load(f"./PanNuke/raw_data/Fold {fold}/images/fold{fold}/types.npy")
        images = np.load(f"./PanNuke/raw_data/Fold {fold}/images/fold{fold}/images.npy")

        save_images_and_masks(images, masks, types, images_dir, masks_dir, tissues_dir, Neoplastic_dir, inflams_dir, Connective_dir, Dead_dir, Epithelial_dir, fold)
        print(f"Fold {fold} processing completed.")
