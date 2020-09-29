
import glob
import os
import random

from torch.utils import data
from torch.utils.data import dataloader
from torchvision import transforms
from PIL import Image

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def _get_imgs_path(img_dir):
    if not os.path.exists(img_dir):
        return None
    return glob.glob(os.path.join(img_dir, "*.jpg"))


def _cat_label(img_path):
    label = os.path.basename(img_path)
    label = '_'.join(label.split('_')[:-1]).lower()
    return label


def _create_labels(img_paths):
    label_set = []
    labels = []
    imgs = []
    for img_path in img_paths:
        label = _cat_label(img_path)
        if label is None:
            continue

        if label not in label_set:
            label_set.append(label)
        imgs.append(img_path)
        labels.append(label)
    return imgs, labels, label_set


def _label_dict(label_set):
    label2ind = {}
    ind2label = {}
    for i, label in enumerate(label_set):
        label2ind[label] = i
        ind2label[i] = label

    return label2ind, ind2label


def _convert_label2inds(labels, label2ind_dict):
    label_idxes = []
    for label in labels:
        if label in label2ind_dict:
            label_idxes.append(label2ind_dict[label])
    return label_idxes


def _load_dataset(img_dir):
    imgs = _get_imgs_path(img_dir)
    # print(imgs)

    img_paths, labels, label_set = _create_labels(imgs)
    label2ind, ind2label = _label_dict(label_set)
    label_idxes = _convert_label2inds(labels, label2ind)

    return img_paths, label_idxes, ind2label


def _data_split(imgs, labels, val_rate=0.1):
    data_idx = list(range(len(imgs)))
    random.shuffle(data_idx)
    new_imgs = []
    new_labels = []
    for img, label in zip(imgs, labels):
        new_imgs.append(img)
        new_labels.append(label)

    train_num = int(len(imgs) * (1 - val_rate))
    train_imgs = new_imgs[:train_num]
    train_labels = new_labels[:train_num]
    val_imgs = new_imgs[train_num:]
    val_labels = new_labels[train_num:]
    return train_imgs, val_imgs, train_labels, val_labels


def _generate_train_transformes(target_size=(32, 32)):
    # http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/#color-jitter-brightness
    randomHorizontalFlip = transforms.RandomHorizontalFlip(p=0.5)
    randomAffin = transforms.RandomAffine(degrees=30, translate=(0.2, 0.2))
    resize = transforms.Resize(target_size, interpolation=Image.NEAREST)

    return transforms.Compose([
        randomHorizontalFlip,
        randomAffin,
        resize,
        transforms.ToTensor(),  # 255で割る
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])


def _generate_val_transformes(target_size=(32, 32)):
    return transforms.Compose([
        transforms.Resize(target_size, interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])


class _PetDataset(data.Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs, self.labels = imgs, labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        img = Image.open(self.imgs[index])
        img = img.convert("RGB")

        label = self.labels[index]

        img = self.transform(img)

        return img, label


def create_dataloader(cnf):
    if not os.path.exists(cnf.img_dir):
        print(f"データセット{cnf.img_dir}は存在しません。")
        return None

    # データセットの読み込み
    imgs, labels, label_set = _load_dataset(cnf.img_dir)
    train_imgs, val_imgs, train_labels, val_labels = _data_split(
        imgs, labels, val_rate=cnf.train.val_rate)

    train_tf = _generate_train_transformes(
        target_size=(cnf.model.input_size, cnf.model.input_size))
    val_tf = _generate_val_transformes(target_size=(
        cnf.model.input_size, cnf.model.input_size))

    train_dataset = _PetDataset(train_imgs, train_labels, transform=train_tf)
    val_dataset = _PetDataset(val_imgs, val_labels, transform=val_tf)

    train_dataloader = data.DataLoader(
        train_dataset, cnf.train.batch_size, shuffle=True,
        num_workers=cnf.train.num_worker,
        pin_memory=cnf.train.pin_memory
    )

    val_dataloader = data.DataLoader(
        val_dataset, cnf.train.batch_size, shuffle=True,
        num_workers=cnf.train.num_worker,
        pin_memory=cnf.train.pin_memory
    )

    dataloaders_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }
    return dataloaders_dict, label_set


if __name__ == "__main__":
    img_dir = "./dataset/images"
    imgs = _get_imgs_path(img_dir)
    # print(imgs)

    img_paths, labels, label_set = _create_labels(imgs)
    print(len(img_paths), len(labels), label_set)
    label2ind, ind2label = _label_dict(label_set)
    label_idxes = _convert_label2inds(labels, label2ind)
    print(len(label_idxes), label_idxes[:2])

    train_imgs, val_imgs, train_labels, val_labels = _data_split(
        img_paths, label_idxes, val_rate=0.2)
    print(len(train_imgs), len(val_imgs))
