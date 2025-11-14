from typing import List, Union

import os
import copy
import numpy as np


import torch
import torchvision
import torchvision.transforms as transforms
from CIFAR import CIFAR100, CIFAR10, CIFAR5, Small_CIFAR10, Small_Binary_CIFAR10, Small_CIFAR5, Small_CIFAR6
from IPython import embed

def manual_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_DATASETS = {}

def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn

def _get_mnist_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_lacuna_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_cifar10_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=4, fill=(125,123,113)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_cifar100_transforms(augment=True):
    transform_augment = transforms.Compose([
            # transforms.Pad(padding=4, fill=(125,123,113)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ]) 
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ]) 
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test





def _get_imagenet_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_mix_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

@_add_dataset   
def cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar10_transforms(augment=augment)
    train_set = CIFAR10(root=root, train=True, transform=transform_train)
    test_set  = CIFAR10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def cifar100(root, augment=False):
    transform_train, transform_test = _get_cifar100_transforms(augment=augment)
    train_set = CIFAR100(root=root, train=True, transform=transform_train)
    test_set  = CIFAR100(root=root, train=False, transform=transform_test)
    return train_set, test_set


@_add_dataset   
def small_cifar5(root, augment=False):
    transform_train, transform_test = _get_cifar10_transforms(augment=augment)
    train_set = Small_CIFAR5(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def cifar5(root, augment=False):
    transform_train, transform_test = _get_cifar10_transforms(augment=augment)
    train_set = CIFAR5(root=root, train=True, transform=transform_train)
    test_set  = CIFAR5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def cifar5(root, augment=False):
    transform_train, transform_test = _get_cifar10_transforms(augment=augment)
    train_set = CIFAR10(root=root, train=True, transform=transform_train)
    test_set  = CIFAR10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar6(root, augment=False):
    transform_train, transform_test = _get_cifar10_transforms(augment=augment)
    train_set = Small_CIFAR6(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR6(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar10_transforms(augment=augment)
    train_set = Small_CIFAR10(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_binary_cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar10_transforms(augment=augment)
    train_set = Small_Binary_CIFAR10(root=root, train=True, transform=transform_train)
    test_set  = Small_Binary_CIFAR10(root=root, train=False, transform=transform_test)
    return train_set, test_set



@_add_dataset
def mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = Small_MNIST(root=root, train=True, transform=transform_train)
    test_set = Small_MNIST(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_binary_mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = Small_Binary_MNIST(root=root, train=True, transform=transform_train)
    test_set = Small_Binary_MNIST(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def lacuna100(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Lacuna100(root=root, train=True, transform=transform_train)
    test_set = Lacuna100(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def lacuna10(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Lacuna10(root=root, train=True, transform=transform_train)
    test_set = Lacuna10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_lacuna5(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Lacuna5(root=root, train=True, transform=transform_train)
    test_set = Small_Lacuna5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_lacuna6(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Lacuna6(root=root, train=True, transform=transform_train)
    test_set = Small_Lacuna6(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_lacuna10(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Lacuna10(root=root, train=True, transform=transform_train)
    test_set = Small_Lacuna10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_binary_lacuna10(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Binary_Lacuna10(root=root, train=True, transform=transform_train)
    test_set = Small_Binary_Lacuna10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_pretrain(root, augment=False):
    transform_train, transform_test = _get_imagenet_transforms(augment=augment)
    train_set = TinyImageNet_pretrain(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_pretrain(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_finetune(root, augment=False):
    transform_train, transform_test = _get_imagenet_transforms(augment=augment)
    train_set = TinyImageNet_finetune(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_finetune(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_finetune5(root, augment=False):
    transform_train, transform_test = _get_imagenet_transforms(augment=augment)
    train_set = TinyImageNet_finetune5(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_finetune5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def mix10(root, augment=False):
    transform_train, transform_test = _get_mix_transforms(augment=augment)
    lacuna_train_set = Lacuna10(root=root, train=True, transform=transform_train)
    lacuna_test_set = Lacuna10(root=root, train=False, transform=transform_test)
    cifar_train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform_train)
    cifar_test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform_test)
    
    lacuna_train_set.targets = np.array(lacuna_train_set.targets)
    lacuna_test_set.targets = np.array(lacuna_test_set.targets)
    cifar_train_set.targets = np.array(cifar_train_set.targets)
    cifar_test_set.targets = np.array(cifar_test_set.targets)
        
    lacuna_train_set.data = lacuna_train_set.data[:,::2,::2,:]
    lacuna_test_set.data = lacuna_test_set.data[:,::2,::2,:]
    
    classes = np.arange(5)
    for c in classes:
        lacuna_train_class_len = np.sum(lacuna_train_set.targets==c)
        lacuna_train_set.data[lacuna_train_set.targets==c]=cifar_train_set.data[cifar_train_set.targets==c]\
                                                            [:lacuna_train_class_len,:,:,:]
        lacuna_test_class_len = np.sum(lacuna_test_set.targets==c)
        lacuna_test_set.data[lacuna_test_set.targets==c]=cifar_test_set.data[cifar_test_set.targets==c]\
                                                            [:lacuna_test_class_len,:,:,:]
    return lacuna_train_set, lacuna_test_set

@_add_dataset
def mix100(root, augment=False):
    transform_train, transform_test = _get_mix_transforms(augment=augment)
    lacuna_train_set = Lacuna100(root=root, train=True, transform=transform_train)
    lacuna_test_set = Lacuna100(root=root, train=False, transform=transform_test)
    cifar_train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=transform_train)
    cifar_test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=transform_test)
    
    lacuna_train_set.targets = np.array(lacuna_train_set.targets)
    lacuna_test_set.targets = np.array(lacuna_test_set.targets)
    cifar_train_set.targets = np.array(cifar_train_set.targets)
    cifar_test_set.targets = np.array(cifar_test_set.targets)
        
    lacuna_train_set.data = lacuna_train_set.data[:,::2,::2,:]
    lacuna_test_set.data = lacuna_test_set.data[:,::2,::2,:]
    
    classes = np.arange(50)    
    for c in classes:
        lacuna_train_class_len = np.sum(lacuna_train_set.targets==c)
        lacuna_train_set.data[lacuna_train_set.targets==c]=cifar_train_set.data[cifar_train_set.targets==c]\
                                                            [:lacuna_train_class_len,:,:,:]
        lacuna_test_class_len = np.sum(lacuna_test_set.targets==c)
        lacuna_test_set.data[lacuna_test_set.targets==c]=cifar_test_set.data[cifar_test_set.targets==c]\
                                                            [:lacuna_test_class_len,:,:,:]
    return lacuna_train_set, lacuna_test_set



from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

#S以下3っつはDNの生成メソッド
def gen_subclass_noise_20(target):
    # One-fifths of the sub-classes are flipped into noisy classes.
    if target in [4, 30, 55, 72, 89]:
        return 0
    elif target in [1, 32, 67, 73, 95]:
        return 1
    elif target in [54, 62, 70, 82, 91]:
        return 2
    elif target in [9, 10, 16, 28, 92]:
        return 3
    elif target in [0, 51, 53, 57, 61]:
        return 4
    elif target in [22, 39, 40, 86, 83]:
        return 5
    elif target in [5, 20, 25, 84, 87]:
        return 6
    elif target in [6, 7, 14, 18, 94]:
        return 7
    elif target in [3, 42, 43, 88, 24]:
        return 8
    elif target in [12, 17, 37, 68, 97]:
        return 9
    elif target in [23, 33, 49, 60, 76]:
        return 10
    elif target in [15, 19, 21, 31, 71]:
        return 11
    elif target in [34, 63, 64, 66, 38]:
        return 12
    elif target in [26, 45, 77, 79, 75]:
        return 13
    elif target in [2, 11, 35, 46, 99]:
        return 14
    elif target in [27, 29, 44, 78, 98]:
        return 15
    elif target in [36, 50, 65, 74, 93]:
        return 16
    elif target in [47, 52, 56, 59, 80]:
        return 17
    elif target in [8, 13, 48, 58, 96]:
        return 18
    elif target in [41, 69, 81, 85, 90]:
        return 19
    return None


def gen_subclass_noise_40(target):
    # Two-fifths of the sub-classes are flipped into noisy classes.
    # Given that CIFAR20-SDN already presents significant challenges, we do not utilize it in our experiments, reserving more complex scenarios for future research.
    if target in [4, 30, 55, 85, 89]:
        return 0
    elif target in [1, 32, 67, 72, 95]:
        return 1
    elif target in [54, 62, 70, 73, 91]:
        return 2
    elif target in [9, 10, 16, 82, 92]:
        return 3
    elif target in [0, 51, 53, 28, 61]:
        return 4
    elif target in [22, 39, 40, 57, 83]:
        return 5
    elif target in [5, 20, 25, 86, 87]:
        return 6
    elif target in [6, 7, 14, 84, 94]:
        return 7
    elif target in [3, 42, 43, 18, 24]:
        return 8
    elif target in [12, 17, 37, 88, 97]:
        return 9
    elif target in [23, 33, 49, 68, 76]:
        return 10
    elif target in [15, 19, 21, 60, 71]:
        return 11
    elif target in [34, 63, 64, 31, 38]:
        return 12
    elif target in [26, 45, 77, 66, 75]:
        return 13
    elif target in [2, 11, 35, 79, 99]:
        return 14
    elif target in [27, 29, 44, 46, 98]:
        return 15
    elif target in [36, 50, 65, 78, 93]:
        return 16
    elif target in [47, 52, 56, 74, 80]:
        return 17
    elif target in [8, 13, 48, 59, 96]:
        return 18
    elif target in [41, 69, 81, 58, 90]:
        return 19
    return None

#正しいスーパークラスを返す
def gen_subclean(target):
    if target in [4, 30, 55, 72, 95]:
        return 0
    elif target in [1, 32, 67, 73, 91]:
        return 1
    elif target in [54, 62, 70, 82, 92]:
        return 2
    elif target in [9, 10, 16, 28, 61]:
        return 3
    elif target in [0, 51, 53, 57, 83]:
        return 4
    elif target in [22, 39, 40, 86, 87]:
        return 5
    elif target in [5, 20, 25, 84, 94]:
        return 6
    elif target in [6, 7, 14, 18, 24]:
        return 7
    elif target in [3, 42, 43, 88, 97]:
        return 8
    elif target in [12, 17, 37, 68, 76]:
        return 9
    elif target in [23, 33, 49, 60, 71]:
        return 10
    elif target in [15, 19, 21, 31, 38]:
        return 11
    elif target in [34, 63, 64, 66, 75]:
        return 12
    elif target in [26, 45, 77, 79, 99]:
        return 13
    elif target in [2, 11, 35, 46, 98]:
        return 14
    elif target in [27, 29, 44, 78, 93]:
        return 15
    elif target in [36, 50, 65, 74, 80]:
        return 16
    elif target in [47, 52, 56, 59, 96]:
        return 17
    elif target in [8, 13, 48, 58, 90]:
        return 18
    elif target in [41, 69, 81, 85, 89]:
        return 19
    return None


def gen_subclass_noise(targets, noise_rate, noise_type=1):
    #noise_typ=1は5つあるサブクラスのうち1種類だけ違うスーパークラスから持ってくる
    #noise_typ=2は5つあるサブクラスのうち2種類だけ違うスーパークラスから持ってくる
    #元論文ではnoise_type=2はむずすぎたから使用してないらしい
    if noise_type == 1:
        # For CIFAR20-SDN, only one-fifth of the classes are flipped.
        # The input noise rate applies to each class, so it's multiplied by five.
        flips = np.random.binomial(1, noise_rate * 5, len(targets))
    else:
        flips = np.random.binomial(1, noise_rate * 5 / 2, len(targets))

    clean_targets = []
    noisy_targets = []
    noise_idx = []
    for i in range(len(targets)):
        clean, noise =0, 0
        clean = gen_subclean(targets[i])
        if flips[i] == 1 and noise_type == 1:
            noise = gen_subclass_noise_20(targets[i])
        elif flips[i] == 1 and noise_type == 2:
            noise = gen_subclass_noise_40(targets[i])
        else:
            noise = gen_subclean(targets[i])
        
        noisy_targets.append(noise)
        clean_targets.append(clean)
        if clean!=noise: noise_idx.append(i)
    return np.array(noisy_targets), np.array(clean_targets), noise_idx



def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='',  pred=[], probability=[], log='', sample_rate=1.0): 
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        manual_seed(0)
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.train_data = test_dic['data']
                self.train_data = self.train_data.reshape((10000, 3, 32, 32))
                self.train_data = self.train_data.transpose((0, 2, 3, 1))  # 転置する関数
                self.noise_label = test_dic['labels']
                self.clean_label = self.noise_label
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.train_data = test_dic['data']
                self.train_data = self.train_data.reshape((10000, 3, 32, 32))
                self.train_data = self.train_data.transpose((0, 2, 3, 1))  
                self.noise_label = test_dic['fine_labels']                            
                self.clean_label = self.noise_label
                if noise_mode=='SDN':
                    clean_targets = [gen_subclean(i) for i in self.clean_label]
                    self.clean_label = clean_targets
                    self.noise_label = self.clean_label

        else:    
            train_data=[]
            train_label=[]
            noise_idx=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))


            #ノイズの生成
            noise_label = []
            noise_idx = []
            # print(noise_file)
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
                if noise_mode=="SDN":_,train_label,_ = gen_subclass_noise(np.array(train_label), self.r)
                noise_idx = [i for i in range(len(noise_label)) if train_label[i]!=noise_label[i]]

            #inject noise
            else:
                if noise_mode=='SDN':
                    noise_label, train_label, noise_idx = gen_subclass_noise(np.array(train_label), self.r)
                else:
                    idx = list(range(50000))
                    random.shuffle(idx)
                    num_noise = int(self.r*50000)            
                    noise_idx = idx[:num_noise]
                    for i in range(50000):
                        if i in noise_idx:
                            if noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)
                            elif noise_mode=='asym':   
                                noiselabel = self.transition[train_label[i]]
                                noise_label.append(noiselabel)  

                        else:    
                            noise_label.append(train_label[i])   
                print("save noisy labels to %s ..."%noise_file) 
            
            if noise_mode=='SDN':noise_label = [int(i) for i in list(noise_label)]
            # print(f'/{self.r:.2f}_{noise_mode}.json')
            # print(f'/{self.r:.1f}_{noise_mode}.json')
            
            path = noise_file.replace(f'/{self.r:.2f}_{noise_mode}.json', "")
            path = path.replace(f'/{self.r:.1f}_{noise_mode}.json', "")
            path = path.replace(f'/noise_file.json', "")
            # print(path)
            os.makedirs(path, exist_ok=True)
            json.dump(list(noise_label),open(noise_file,"w"))
        
            if self.mode == 'all':
                # print(sum(noise_label==train_label))
                self.train_data = train_data
                self.noise_label = noise_label
                self.clean_label = train_label

            elif self.mode == 'noise_pred':
                self.train_data = train_data
                self.noise_label = noise_label
                self.clean_label = train_label
                self.c_or_n = (np.array(noise_label)==np.array(train_label))

            elif self.mode == 'retain':
                retain_idx = list(set(range(50000)) - set(noise_idx))
                self.train_data = train_data[retain_idx]
                self.noise_label = np.array(noise_label)[np.array(retain_idx)]
                self.clean_label = np.array(train_label)[np.array(retain_idx)]

            elif self.mode == 'forget':
                forget_idx=noise_idx
                self.train_data = train_data[forget_idx]
                self.noise_label = np.array(noise_label)[forget_idx]
                self.clean_label = np.array(train_label)[forget_idx]
            
            elif self.mode == 'rand-sampling-f':
                forget_idx = noise_idx
                rand_forget_idx = np.random.choice(forget_idx, int(len(forget_idx)*sample_rate), replace=False)
                retain_idx = list(set(range(50000)) - set(noise_idx))
                rand_retain_idx = np.random.choice(retain_idx, int(len(retain_idx)*sample_rate), replace=False)
                rand_forget_idx = list(rand_forget_idx)+list(set(retain_idx)-set(rand_retain_idx))
                self.train_data = train_data[rand_forget_idx]
                self.noise_label = np.array(noise_label)[rand_forget_idx]
                self.clean_label = np.array(train_label)[rand_forget_idx]

            elif self.mode == 'rand-sampling-r':
                forget_idx = noise_idx
                rand_forget_idx = np.random.choice(forget_idx, int(len(forget_idx)*sample_rate), replace=False)
                retain_idx = list(set(range(50000)) - set(noise_idx))
                rand_retain_idx = np.random.choice(retain_idx, int(len(retain_idx)*sample_rate), replace=False)
                rand_retain_idx = list(rand_retain_idx)+list(set(forget_idx)-set(rand_forget_idx))
                self.train_data = train_data[rand_retain_idx]
                self.noise_label = np.array(noise_label)[rand_retain_idx]
                self.clean_label = np.array(train_label)[rand_retain_idx]

            
            elif self.mode == 're-retain':
                retain_idx = pred.nonzero()[0]
                self.train_data = train_data[retain_idx]
                self.noise_label = np.array(noise_label)[np.array(retain_idx)]
                self.clean_label = np.array(train_label)[np.array(retain_idx)]

            elif self.mode == 're-forget':
                forget_idx = list(set(range(50000))-set(pred.nonzero()[0]))
                self.train_data = train_data[forget_idx]
                self.noise_label = np.array(noise_label)[forget_idx]
                self.clean_label = np.array(train_label)[forget_idx]

            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0] # 要素が０以外のインデックスを取得
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]       # predはFalse, True                                        
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))    
                
    def __getitem__(self, index):
        if self.mode == 'noise_pred':
            img, target, clean, c_or_n= self.train_data[index], self.noise_label[index], self.clean_label[index], self.c_or_n[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, clean, c_or_n

        else:
            img, target, clean= self.train_data[index], self.noise_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, clean       

           
    def __len__(self):
        return len(self.train_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, noise_file='', retain_bs=32, forget_bs=32, sample_rate=0.5):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.retain_bs = retain_bs
        self.forget_bs = forget_bs
        self.sample_rate = sample_rate
        # transforms.Pad(padding=4, fill=(125,123,113)),scrubを参考に左を追加
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    # transforms.Pad(padding=4, fill=(125,123,113)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    # transforms.Pad(padding=4, fill=(125,123,113)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,pred=[],prob=[], shuffle = True):
        if mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size, 
                shuffle=shuffle,
                num_workers=self.num_workers)          
            return eval_loader
        
        elif mode=='noise_pred':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode='noise_pred', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size, 
                shuffle=shuffle,
                num_workers=self.num_workers)          
            return eval_loader
        
        elif mode == 'retain':
            confuse_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode='retain', noise_file=self.noise_file)      
            confuse_loader = DataLoader(
                dataset=confuse_dataset, 
                batch_size=self.retain_bs, 
                shuffle=shuffle,
                num_workers=self.num_workers)  
            return confuse_loader
        
        elif mode == 'forget':
            confuse_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode='forget', noise_file=self.noise_file)      
            confuse_loader = DataLoader(
                dataset=confuse_dataset, 
                batch_size=self.forget_bs, 
                shuffle=shuffle,
                num_workers=self.num_workers)  
            return confuse_loader
        
        elif mode =='re-retain':
            confuse_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode='re-retain', noise_file=self.noise_file, pred=pred)      
            confuse_loader = DataLoader(
                dataset=confuse_dataset, 
                batch_size=self.retain_bs, 
                shuffle=shuffle,
                num_workers=self.num_workers)  
            return confuse_loader
        
        elif mode =='re-forget':
            confuse_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode='re-forget', noise_file=self.noise_file, pred=pred)      
            confuse_loader = DataLoader(
                dataset=confuse_dataset, 
                batch_size=self.forget_bs, 
                shuffle=shuffle,
                num_workers=self.num_workers)  
            return confuse_loader

        elif mode == "rand-sampling":
            retain_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode=mode+'-r', noise_file=self.noise_file, pred=pred, sample_rate=self.sample_rate)      
            retain_loader = DataLoader(
                dataset=retain_dataset, 
                batch_size=self.retain_bs, 
                shuffle=shuffle,
                num_workers=self.num_workers)  
            forget_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode=mode+'-f', noise_file=self.noise_file, pred=pred, sample_rate=self.sample_rate)      
            forget_loader = DataLoader(
                dataset=retain_dataset, 
                batch_size=self.forget_bs, 
                shuffle=shuffle,
                num_workers=self.num_workers)  
            print(f"rand-sampling: retain {len(retain_dataset)} forget {len(forget_dataset)}")
            return forget_loader, retain_loader