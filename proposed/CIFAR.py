import numpy as np
import os
from PIL import Image
import torchvision
from torchvision.datasets import VisionDataset
root = os.path.expanduser('~/data')

np.random.seed(0)



class CIFAR10(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CIFAR10, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        ds.targets=np.array(ds.targets)
        # print('in_CIFAR10')
        self.targets=ds.targets
        self.data=ds.data
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, clean_labels = self.data[index], self.targets[index], self.clean_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            clean_labels = self.target_transform(clean_labels)

        return img, target, clean_labels

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")



class CIFAR100(VisionDataset):
    # """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    # This is a subclass of the `CIFAR10` Dataset.
    # """
    # base_folder = 'cifar-100-python'
    # url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    # filename = "cifar-100-python.tar.gz"
    # tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    # train_list = [
    #     ['train', '16019d7e3df5f24257cddd939b257f8d'],
    # ]

    # test_list = [
    #     ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    # ]
    # meta = {
    #     'filename': 'meta',
    #     'key': 'fine_label_names',
    #     'md5': '7973b15100ade9c7d40fb424638fde48',
    # }

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CIFAR100, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR100(root=root, train=train, download=True)
        ds.targets=np.array(ds.targets)
        print('in_CIFAR100')
        self.targets=ds.targets
        self.data=ds.data
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, clean_labels = self.data[index], self.targets[index], self.clean_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            clean_labels = self.target_transform(clean_labels)

        return img, target, clean_labels

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")





class Small_CIFAR10(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_CIFAR10, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        ds.targets=np.array(ds.targets)
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(10):
            if self.train:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],125,replace=False)
            else:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],100,replace=False)
                #np.where(ds.targets==i)[0]
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])
            sub_ds_target_list.append(ds.targets[sub_cls_id])
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, clean_labels = self.data[index], self.targets[index], self.clean_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            clean_label = self.target_transform(clean_label)

        return img, target, clean_labels

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class Small_CIFAR5(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_CIFAR5, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        ds.targets=np.array(ds.targets)
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(5):
            if self.train:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],125,replace=False)
            else:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],100,replace=False)
                #np.where(ds.targets==i)[0]                
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])
            sub_ds_target_list.append(ds.targets[sub_cls_id])
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target, clean_label = self.data[index], self.targets[index], self.clean_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            clean_label = self.target_transform(clean_label)

        return img, target, clean_label

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR5(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CIFAR5, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        ds.targets=np.array(ds.targets)
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(5):
            if self.train:
                sub_cls_id = np.where(ds.targets==i)[0]
            else:
                sub_cls_id = np.where(ds.targets==i)[0]
                #np.where(ds.targets==i)[0]                
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])
            sub_ds_target_list.append(ds.targets[sub_cls_id])
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target, clean_label = self.data[index], self.targets[index], self.clean_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            clean_label = self.target_transform(clean_label)

        return img, target, clean_label

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class Small_CIFAR6(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_CIFAR6, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        ds.targets=np.array(ds.targets)
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(6):
            if self.train:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],125,replace=False)
            else:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],100,replace=False)
                #np.where(ds.targets==i)[0]                
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])
            sub_ds_target_list.append(ds.targets[sub_cls_id])
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class Small_Binary_CIFAR10(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_Binary_CIFAR10, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        ds.targets=np.array(ds.targets)
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(2):
            if self.train:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],250,replace=False)
            else:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],250,replace=False)
                #np.where(ds.targets==i)[0]                
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])
            sub_ds_target_list.append(ds.targets[sub_cls_id])
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
