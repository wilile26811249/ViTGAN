from image_dataset import CrypkoDataset, InfiniteSampler

import os
import glob
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

def get_dataset(root, img_size = 64):
    fnames = glob.glob(os.path.join(root, '*'))
    compose = [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
    ]
    transform = T.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset


def get_dataloader(root, img_size = 64, batch_size = 64):
    data = get_dataset(root, img_size)
    data_loader = iter(
        DataLoader(
            data,
            batch_size = batch_size,
            num_workers = 1,
            sampler = InfiniteSampler(data)
        )
    )
    return data_loader


def test(root_dir = "crypko_data/faces/"):
    dataset = get_dataset(root = root_dir)
    print(dataset[1].shape)
    images = [dataset[i] for i in range(26, 42)]
    grid_img = torchvision.utils.make_grid(images, nrow = 4)
    torchvision.utils.save_image(grid_img, "sample.jpg")
    print("Save image to sample.jpg from CryphoDataset.")
