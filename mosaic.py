import csv
import numpy as np
import cv2
from torchvision import datasets, transforms
from skimage.color import rgb2lab, lab2rgb #pip install scikit-image


DATA_ROOT = "./data"
AVERAGES_CSV = "averages.csv"

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


def make_mosiac(in_path, out_path, block_size=32):

    #load datasets
    train, test = load_images()
    indices, means = load_averages()

    #Load the Base Image and compute number of blocks
    base_img = cv2.imread(in_path)

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
            print (f"Placing block ({by}, {bx}) with image index {closest_idx}")

            closest_img = load_image_by_index(closest_idx, train, test)


            #Resize and place the image in the output mosaic
            resized_img = cv2.resize(closest_img, (block_size, block_size))
            out_img[y1:y2, x1:x2, :] = resized_img

    #Convert back to BGR and save
    cv2.imwrite(out_path, out_img)
    print(f"Mosaic saved to {out_path}")

if __name__ == "__main__":
    make_mosiac("Input_Images/flower.jpg", "mosaic_output.jpg", block_size=32)

    

