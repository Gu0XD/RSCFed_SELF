# encoding: utf-8
"""
Read images and corresponding labels.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST
N_CLASSES = 10

class CheXpertDataset(Dataset):
    def __init__(self, dataset_type, data_np, label_np, pre_w, pre_h, lab_trans=None, un_trans_wk=None, data_idxs=None,
                 is_labeled=False,
                 is_testing=False):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CheXpertDataset, self).__init__()

        self.images = data_np
        self.labels = label_np
        self.is_labeled = is_labeled
        self.dataset_type = dataset_type
        self.is_testing = is_testing

        self.resize = transforms.Compose([transforms.Resize((pre_w, pre_h))])
        if not is_testing:
            if is_labeled == True:
                self.transform = lab_trans
            else:
                self.data_idxs = data_idxs
                self.weak_trans = un_trans_wk
        else:
            self.transform = lab_trans

        print('Total # images:{}, labels:{}'.format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        if self.dataset_type == 'skin':
            img_path = self.images[index]
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.fromarray(self.images[index]).convert('RGB')

        image_resized = self.resize(image)
        label = self.labels[index]

        if not self.is_testing:
            if self.is_labeled == True:
                if self.transform is not None:
                    image = self.transform(image_resized).squeeze()
                    # image=image[:,:224,:224]
                    return index, image, torch.FloatTensor([label])
            else:
                if self.weak_trans and self.data_idxs is not None:
                    weak_aug = self.weak_trans(image_resized)
                    idx_in_all = self.data_idxs[index]

                    for idx in range(len(weak_aug)):
                        weak_aug[idx] = weak_aug[idx].squeeze()
                    return index, weak_aug, torch.FloatTensor([label])
        else:
            image = self.transform(image_resized)
            return index, image, torch.FloatTensor([label])
            # return index, weak_aug, strong_aug, torch.FloatTensor([label])

    def __len__(self):
        return len(self.labels)

class Generated(MNIST):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        self.dataidxs = dataidxs

        if self.train:
            self.data = np.load("data/generated/X_train.npy")
            self.targets = np.load("data/generated/y_train.npy")
        else:
            self.data = np.load("data/generated/X_test.npy")
            self.targets = np.load("data/generated/y_test.npy")            

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]        


    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self):
        return len(self.data)

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return [out1, out2]