# PanNuke_NoNuSeg_cpm17_ConSep_Preprocess

A repository for preprocessing widely used H&E image datasets, including PanNuke, MoNuSeg, CPM17, and ConSep, to facilitate nuclei segmentation tasks with deep learning models.

## Datasets

### PanNuke
PanNuke is an H&E stained image dataset containing 7,904 256 × 256 patches from 19 different tissue types. The nuclei are classified into the following categories:
- Neoplastic
- Inflammatory
- Connective/Soft Tissue
- Dead
- Epithelial Cells

The dataset is divided into three folds for use in cross-validation. You can download the dataset [here](https://jgamper.github.io/PanNukeDataset/).

#### Prepare the Data

After downloading the PanNuke dataset, you will receive three `.zip` files named `fold1.zip`, `fold2.zip`, and `fold3.zip`. These files contain the data stored as NumPy arrays. Once extracted, the directory structure will be as follows:

```
📦Fold 1
 ┣ 📂images
 ┃ ┗ 📂fold1
 ┃ ┃ ┣ 📜images.npy
 ┃ ┃ ┗ 📜types.npy
 ┣ 📂masks
 ┃ ┣ 📂fold1
 ┃ ┃ ┗ 📜masks.npy
 ┃ ┣ 📜by-nc-sa.md
 ┃ ┗ 📜README.md
 ┗ 📜README.md
```
Fold 2 and 3 also have similar structure

how to run the preprocess code:

```bash
python3 pannuke_process.py
```
The following is a sample of the preprocessed output:
![Sample Output](./pics/output_pannauke.png)

### Instructions for Preparation
1. Download the `.zip` files from the [official PanNuke dataset page](https://jgamper.github.io/PanNukeDataset/).
2. Extract each `.zip` file into a separate folder (`Fold 1`, `Fold 2`, `Fold 3`).
3. Verify the directory structure as shown above.

You are now ready to preprocess the data using this repository.


### MoNuSeg
[Add description here about MoNuSeg, including key characteristics and use cases.]

### CPM17
[Add description here about CPM17, including its focus and unique features.]

### ConSep
[Add description here about ConSep, highlighting its structure and relevance for nuclei segmentation.]
