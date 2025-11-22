import os
import glob
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from pathlib import Path
import math

# MONAI Imports
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, CropForegroundd, 
    ResizeWithPadOrCropd
)
from monai.data import Dataset, DataLoader


# 1. Configuration

class TestConfig:
    SPATIAL_SIZE = (64, 64, 64)
    IN_CHANNELS = 1
    OUT_CHANNELS = 49
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')

# Label Names
LABEL_NAMES = {
    0: "Background",
    1: "Lower Jawbone", 2: "Upper Jawbone",
    3: "Left Inf. Alveolar Canal", 4: "Right Inf. Alveolar Canal",
    5: "Left Maxillary Sinus", 6: "Right Maxillary Sinus",
    7: "Pharynx", 8: "Bridge", 9: "Crown", 10: "Implant",
    11: "UR Central Incisor", 12: "UR Lateral Incisor",
    13: "UR Canine", 14: "UR 1st Premolar",
    15: "UR 2nd Premolar", 16: "UR 1st Molar",
    17: "UR 2nd Molar", 18: "UR 3rd Molar",
    21: "UL Central Incisor", 22: "UL Lateral Incisor",
    23: "UL Canine", 24: "UL 1st Premolar",
    25: "UL 2nd Premolar", 26: "UL 1st Molar",
    27: "UL 2nd Molar", 28: "UL 3rd Molar",
    31: "LL Central Incisor", 32: "LL Lateral Incisor",
    33: "LL Canine", 34: "LL 1st Premolar",
    35: "LL 2nd Premolar", 36: "LL 1st Molar",
    37: "LL 2nd Molar", 38: "LL 3rd Molar",
    41: "LR Central Incisor", 42: "LR Lateral Incisor",
    43: "LR Canine", 44: "LR 1st Premolar",
    45: "LR 2nd Premolar", 46: "LR 1st Molar",
    47: "LR 2nd Molar", 48: "LR 3rd Molar"
}


# 2. Visualization: The Montage (Grid) Function

def save_all_slices_montage(image_vol, mask_vol, filename, output_dir):
    """
    Creates an 8x8 grid showing ALL 64 slices of the volume (Axial View).
    """
    plot_dir = os.path.join(output_dir, "plots")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    # We iterate over Depth (Axial slices)
    depth = image_vol.shape[2] # Assuming RAS orientation, Z is usually index 2
    
    # Grid dimensions: 8x8 = 64 slices
    rows = 8
    cols = 8
    
    # Create a large canvas
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    fig.suptitle(f"Full Volume Montage: {filename}\n(Slice 0 top-left to Slice 63 bottom-right)", fontsize=20)

    # Colormap Setup
    cmap = plt.get_cmap("nipy_spectral")
    norm = plt.Normalize(vmin=0, vmax=49)

    # Identify classes present in the WHOLE volume for the legend
    present_classes = np.unique(mask_vol)
    present_classes = present_classes[present_classes != 0]

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    for i in range(rows * cols):
        ax = axes_flat[i]
        
        if i < depth:
            # Get the slice (Axial view)
            # Note: Depending on your data orientation, you might need to switch indices.
            # Usually index 2 is Z (Axial) in RAS.
            img_slice = np.rot90(image_vol[:, :, i]) 
            mask_slice = np.rot90(mask_vol[:, :, i])
            
            # Plot Image
            ax.imshow(img_slice, cmap="gray", origin="lower")
            
            # Plot Mask Overlay
            if np.sum(mask_slice) > 0: # Only overlay if mask exists on this slice
                masked_label = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax.imshow(masked_label, cmap=cmap, norm=norm, alpha=0.5, origin="lower")
            
            ax.set_title(f"Slice {i}", fontsize=8)
            ax.axis("off")
        else:
            # Hide unused subplots if any
            ax.axis("off")

    # --- Dynamic Legend (Outside the grid) ---
    if len(present_classes) > 0:
        patches = []
        for cls_idx in sorted(present_classes):
            color = cmap(norm(cls_idx))
            label_text = f"{int(cls_idx)}: {LABEL_NAMES.get(int(cls_idx), 'Unknown')}"
            patch = mpatches.Patch(color=color, label=label_text)
            patches.append(patch)
        
        # Place legend on the right side
        fig.legend(handles=patches, loc='center right', 
                   bbox_to_anchor=(1.1, 0.5), 
                   title="Detected Classes",
                   fontsize='medium', title_fontsize='large')

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"{filename}_montage.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# 3. Main Inference Function

def run_inference(model_path, image_dir, output_dir="./inference_results"):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading model from: {model_path}")
    
    try:
        model = torch.load(model_path, map_location=TestConfig.DEVICE, weights_only=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    model.eval()
    model.to(TestConfig.DEVICE)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.mha"))) + \
                  sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    
    if not image_paths:
        print("No images found!")
        return None

    print(f"Found {len(image_paths)} images.")
    for p in image_paths:
        print(f"   -> {os.path.basename(p)}")

    # Transforms
    test_transforms = Compose([
        LoadImaged(keys=["image"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear"), allow_missing_keys=True),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=3000, b_min=0.0, b_max=1.0),
        CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > 0),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=TestConfig.SPATIAL_SIZE, mode="constant"),
    ])

    # Create Dataset & Loader
    data_dicts = [{"image": img, "name": os.path.basename(img)} for img in image_paths]
    ds = Dataset(data=data_dicts, transform=test_transforms)
    loader = DataLoader(ds, batch_size=60, shuffle=False, num_workers=0)

    # Inference Loop
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(loader, desc="Processing Volumes")):
            
            images = batch_data["image"].to(TestConfig.DEVICE)
            filename = batch_data["name"][0]
            
            # --- PREDICTION ---
            outputs = model(images)
            pred_tensor = torch.argmax(outputs, dim=1).detach().cpu()
            
            # --- SAVE NIFTI ---
            output_path = os.path.join(output_dir, f"pred_{filename}")
            if output_path.endswith('.mha'): output_path = output_path.replace('.mha', '.nii.gz')
            
            affine = np.eye(4)
            if "image_meta_dict" in batch_data:
                 affine = batch_data["image_meta_dict"]["affine"][0].numpy()
            
            pred_img = nib.Nifti1Image(pred_tensor[0].numpy().astype(np.uint8), affine)
            nib.save(pred_img, output_path)

            # --- SAVE MONTAGE PLOT ---
            # Squeeze dimensions: (1, 1, 64, 64, 64) -> (64, 64, 64)
            img_vol_np = images.detach().cpu().numpy()[0, 0, :, :, :]
            mask_vol_np = pred_tensor.numpy()[0, :, :, :]
            
            save_all_slices_montage(img_vol_np, mask_vol_np, filename, output_dir)

    print(f"\nProcessing complete! Check the '{output_dir}/plots' folder.")


# Runner

if __name__ == "__main__":
    # UPDATE THESE PATHS
    MODEL_PATH = "/Users/shomer/Desktop/python_test_project/3d_medical_segmentation/best1.pth"
    TEST_IMG_DIR = "/Users/shomer/Desktop/python_test_project/3d_medical_segmentation/test_sample/imagesTr"
    OUTPUT_DIR = "./inference_results"

    run_inference(MODEL_PATH, TEST_IMG_DIR, OUTPUT_DIR)