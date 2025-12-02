import csv
import numpy as np
from torchvision import datasets, transforms
from skimage.color import rgb2lab, lab2rgb #pip install scikit-image


DATA_ROOT = "./data"
OUT_CSV = "averages.csv"

def load_cifar10():

    #Load the sataset
    tf = transforms.ToTensor()
    train_set = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=tf)


    with open(OUT_CSV, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["index", "L_mean", "A_mean", "B_mean"])

        for dataset in [train_set, test_set]:

            for i, (img, _) in enumerate(dataset):
                img_np = img.numpy().transpose((1, 2, 0))  # Convert to HWC format
                lab_img = rgb2lab(img_np)

                L_mean = np.mean(lab_img[:, :, 0])
                A_mean = np.mean(lab_img[:, :, 1])
                B_mean = np.mean(lab_img[:, :, 2])

                writer.writerow([i, L_mean, A_mean, B_mean])
    print(f"Averages saved to {OUT_CSV}")

if __name__ == "__main__":
    load_cifar10()


