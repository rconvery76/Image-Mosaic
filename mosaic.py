import csv
import os
import numpy as np
import cv2
import math
from torchvision import datasets, transforms
from skimage.color import rgb2lab, lab2rgb 
import math          
import matplotlib.pyplot as plt

DATA_ROOT = "./data"
AVERAGES_CSV = "averages.csv"
OUTPUT_DIR = "Output_Images"

def load_images():

    train = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transforms.ToTensor())

    return train, test

def load_averages():
    indices = []
    means = []
    with open(AVERAGES_CSV, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            indices.append(int(row["index"]))
            L = float(row["L_mean"])
            A = float(row["A_mean"])
            B = float(row["B_mean"])
            means.append((L, A, B))

    return indices, means

def find_closest_image(L_mean, A_mean, B_mean, means):
    closest_index = -1
    closest_distance = float('inf')

    for i, (L, A, B) in enumerate(means):
        distance = (L - L_mean) ** 2 + (A - A_mean) ** 2 + (B - B_mean) ** 2
        if distance < closest_distance:
            closest_distance = distance
            closest_index = i

    return closest_index

def load_image_by_index(index, train, test):
    if index < len(train):
        img, _ = train[index]
    else:
        img, _ = test[index - len(train)]
    img_np = img.numpy().transpose((1, 2, 0))  # Convert to HWC format
    img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img_bgr

def compute_mse_psnr(orig_bgr, recon_bgr):
    orig = orig_bgr.astype(np.float32)
    recon = recon_bgr.astype(np.float32)

    diff = orig - recon
    mse = np.mean(diff ** 2)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * math.log10((255.0 ** 2) / mse)

    return mse, psnr

def make_mosiac(in_path, out_path, block_size=32):

    #load datasets
    train, test = load_images()
    indices, means = load_averages()

    #Load the Base Image and compute number of blocks
    base_img = cv2.imread(in_path)
    if base_img is None:
        raise FileNotFoundError(f"Could not read image at path: {in_path}")

    h, w, _ = base_img.shape
    h_blocks = h // block_size
    w_blocks = w // block_size

    new_h = h_blocks * block_size
    new_w = w_blocks * block_size

    h_blocks = new_h // block_size
    w_blocks = new_w // block_size

    base_img = cv2.resize(base_img, (new_w, new_h))
    base_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    base_lab = rgb2lab(base_rgb.astype(np.float32) / 255.0)

    #create output image
    out_img = np.zeros_like(base_img, dtype=np.uint8)


    for by in range(h_blocks):
        for bx in range(w_blocks):

            #Get the block
            y1 = by * block_size
            y2 = y1 + block_size
            x1 = bx * block_size
            x2 = x1 + block_size

            block = base_lab[y1:y2, x1:x2, :]

            #Compute the mean LAB values of the block
            L_mean = np.mean(block[:, :, 0])
            A_mean = np.mean(block[:, :, 1])
            B_mean = np.mean(block[:, :, 2])

            #Find the closest matching image from averages.csv
            closest_idx = find_closest_image(L_mean, A_mean, B_mean, means)
            #print (f"Placing block ({by}, {bx}) with image index {closest_idx}")

            closest_img = load_image_by_index(closest_idx, train, test)


            #Resize and place the image in the output mosaic
            resized_img = cv2.resize(closest_img, (block_size, block_size))
            out_img[y1:y2, x1:x2, :] = resized_img

    #Compute MSE and PSNR
    mse, psnr = compute_mse_psnr(base_img, out_img)
    print(f"Mosaic saved to {out_path}")
    print(f"MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")

    # save mosaic
    cv2.imwrite(out_path, out_img)

    # return metrics so we can log them
    return mse, psnr

import numpy as np

def plot_results(results):
    first_key = next(iter(results))
    block_sizes = results[first_key]["block_sizes"]
    x = np.arange(len(block_sizes))  # [0, 1, 2]

    name_map = {
        "flower": "Flower",
        "img2": "Landscape",
        "img3": "Bird",
    }

    # MSE 
    plt.figure()
    for name, metrics in results.items():
        mse = metrics["mse"]
        plt.plot(x, mse, marker="o", label=name_map.get(name, name))

    plt.xlabel("Block size")
    plt.ylabel("MSE")
    plt.title("MSE vs block size")
    plt.xticks(x, block_sizes)  
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mse_calc.png"))

    # PSNR 
    plt.figure()
    for name, metrics in results.items():
        psnr = metrics["psnr"]
        plt.plot(x, psnr, marker="o", label=name_map.get(name, name))

    plt.xlabel("Block size")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs block size")
    plt.xticks(x, block_sizes)   
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "psnr_calc.png"))

if __name__ == "__main__":
    # Image paths
    img1_path = "Input_Images/flower.jpg"
    img2_path = "Input_Images/landscape.jpg"
    img3_path = "Input_Images/bird.jpg"

    # Image 1
    mse_flower_16, psnr_flower_16 = make_mosiac(
        img1_path, os.path.join(OUTPUT_DIR, "mosaic_flower_16.jpg"), block_size=16
    )
    mse_flower_32, psnr_flower_32 = make_mosiac(
        img1_path, os.path.join(OUTPUT_DIR, "mosaic_flower_32.jpg"), block_size=32
    )
    mse_flower_64, psnr_flower_64 = make_mosiac(
        img1_path, os.path.join(OUTPUT_DIR, "mosaic_flower_64.jpg"), block_size=64
    )

    # Image 2
    mse_img2_16, psnr_img2_16 = make_mosiac(
        img2_path, os.path.join(OUTPUT_DIR, "mosaic_landscape_16.jpg"), block_size=16
    )
    mse_img2_32, psnr_img2_32 = make_mosiac(
        img2_path, os.path.join(OUTPUT_DIR, "mosaic_landscape_32.jpg"), block_size=32
    )
    mse_img2_64, psnr_img2_64 = make_mosiac(
        img2_path, os.path.join(OUTPUT_DIR, "mosaic_landscape_64.jpg"), block_size=64
    )

    # Image 3
    mse_img3_16, psnr_img3_16 = make_mosiac(
        img3_path, os.path.join(OUTPUT_DIR, "mosaic_bird_16.jpg"), block_size=16
    )
    mse_img3_32, psnr_img3_32 = make_mosiac(
        img3_path, os.path.join(OUTPUT_DIR, "mosaic_bird_32.jpg"), block_size=32
    )
    mse_img3_64, psnr_img3_64 = make_mosiac(
        img3_path, os.path.join(OUTPUT_DIR, "mosaic_bird_64.jpg"), block_size=64
    )


    # Format and plot results
    results = {
        "flower": {
            "block_sizes": [16, 32, 64],
            "mse":  [mse_flower_16, mse_flower_32, mse_flower_64],
            "psnr": [psnr_flower_16, psnr_flower_32, psnr_flower_64],
        },
        "img2": {
            "block_sizes": [16, 32, 64],
            "mse":  [mse_img2_16, mse_img2_32, mse_img2_64],
            "psnr": [psnr_img2_16, psnr_img2_32, psnr_img2_64],
        },
        "img3": {
            "block_sizes": [16, 32, 64],
            "mse":  [mse_img3_16, mse_img3_32, mse_img3_64],
            "psnr": [psnr_img3_16, psnr_img3_32, psnr_img3_64],
        },
    }

    plot_results(results)

    

