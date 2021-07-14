from image_dataset import CrypkoDataset

import os
import glob
import torchvision
import torchvision.transforms as T

def get_dataset(root, img_size = 64):
    fnames = glob.glob(os.path.join(root, '*'))
    # 1. Resize the image to (64, 64)
    # 2. Linearly map [0, 1] to [-1, 1]
    compose = [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
    ]
    transform = T.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset


def test(root_dir = "crypko_data/faces/"):
    dataset = get_dataset(root = root_dir)
    print(dataset[1].shape)
    images = [dataset[i] for i in range(26, 42)]
    grid_img = torchvision.utils.make_grid(images, nrow = 4)
    torchvision.utils.save_image(grid_img, "sample.jpg")
    print("Save image to sample.jpg from CryphoDataset.")
