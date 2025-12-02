import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from itertools import cycle
import os
import time
import math
import pandas as pd
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
    
from utils import *
import copy
import torch.nn as nn
from torch.autograd import Variable
from typing import List
import itertools
from tqdm.autonotebook import tqdm
from models import *
import models
from logger import *
import wandb
import sys
from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss

from thirdparty.repdistiller.helper.loops import train_forget, train_distill, validate
from thirdparty.repdistiller.helper.pretrain import init
import warnings

# 警告を非表示にする
warnings.filterwarnings("ignore", category=UserWarning)

parser=argparse.ArgumentParser()
parser.add_argument('--model', default='preactresnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--noise_rate', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--num_to_forget', type=int, default=None,
                        help='Number of samples of class to forget')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
parser.add_argument('--pred', type=str, default='', choices=['gmm', 'cls','k-means'], help='how to predict noise-sample')
parser.add_argument('--gpu', default=0, type=int, help='use gpu-id')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--gamma', type=float, default=1, help='cross entropy in minimize')
parser.add_argument('--beta', type=float, default=0, help='cross entropy in minimize')
parser.add_argument('--alpha', type=float, default=0, help='cross entropy in minimize')
parser.add_argument('--delta', type=float, default=0, help='cos_sim in minimize')
parser.add_argument('--zeta', type=float, default=0, help='cos_sim in maximize')
parser.add_argument('--eta', type=float, default=1, help='loss_div in maximize')
parser.add_argument('--epochs', type=int, default=200, help='learning epoch of pre-train model')
parser.add_argument('--noise_mode', type=str, default='sym', choices=['sym', 'asym', 'SDN'], help='asym or sym or SDN(Subclass Domain Noise)')
parser.add_argument('--method', type=str, default='scrub')
parser.add_argument('--e_n', nargs='+', type=int, default=[5])
parser.add_argument('--e_r', nargs='*', type=int, default=[15])
parser.add_argument('--forget_bs', type=int, default=512)
parser.add_argument('--retain_bs', type=int, default=128)
parser.add_argument('--kd_T', type=float, default=0.5, help='kd_loss parameter (loss * args.kd_T**)')
#parameter of t-SNE
parser.add_argument('--tsne_lr', type=int, default=350)
parser.add_argument('--tsne_per', type=int, default=30)
args = parser.parse_args()

#seedの固定
os.environ['PYTHONHASEED']=str(args.seed)
torch.backends.cudnn.deterministic = True
# Faleseにしないと再現性が落ちる
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if use_cuda else "cpu")
file_name=f'{args.model}_{args.dataset}_{args.noise_rate}_{args.noise_mode}'
#クラス数とデータパス，ノイズ率の設定
if 'cifar5' in args.dataset:
	args.num_classes=5
elif args.dataset=='cifar100':
    args.num_classes=100
elif 'cifar10' in args.dataset:
	args.num_classes=10
args.forget_class=list(range(args.num_classes))
args.dataroot='../../data/cifar100/cifar-100-python' if args.dataset=='cifar100' else '../../data/cifar10/cifar-10-batches-py'
if args.num_to_forget!=None:
    args.noise_rate=float(args.num_to_forget/50000)
elif args.noise_rate!=None:
    args.num_to_forget=int(50000*args.noise_rate)
else:
    raise ValueError("both num_to_forget and noise_rate!")

if args.method=="scrub":
    args.gamma=1
    args.alpha=0.01
    args.beta=0
    args.delta=0
    args.eta=0.5
    args.zeta=0
    # args.kd_T=1.
    args.forget_bs=512
    args.retain_bs=128
    args.e_n=[5]
    args.e_r=[15]
    print('--in==scrub')
noise_rate_name=str(args.noise_rate).replace('.', '_')
dir_name = f'./net/per_{args.tsne_per}/lr_{args.tsne_lr}/pretrain/'
os.makedirs(dir_name, exist_ok=True)

args.dir_name = dir_name
torch.cuda.set_device(args.gpu)
def create_model():
    num_classes = 20 if args.noise_mode=='SDN' and args.dataset=='cifar100' else args.num_classes
    temp_model = models.get_model(args.model, num_classes=num_classes)
    temp_model = temp_model.cuda()
    return temp_model

checkpoint = torch.load(f'./weight_save_15.tar')
model = create_model()
model.load_state_dict(checkpoint['state_dict'])

noise_file = f'../../weight/{args.model}_{args.dataset}_{args.noise_rate}_{args.noise_mode}/net_seed{args.seed}/{args.noise_rate:.2f}_{args.noise_mode}.json'
loader = datasets.cifar_dataloader(args.dataset,r=args.noise_rate,noise_mode=args.noise_mode,
                                     batch_size=args.batch_size,num_workers=12,\
    root_dir=args.dataroot,noise_file=noise_file,\
        retain_bs=args.retain_bs, forget_bs=args.forget_bs)
from collections import Counter
train_loader = loader.run('eval_train')
test_loader = loader.run('test')
correct_retain_loader = loader.run('retain')
eval_retain_loader = loader.run('retain', shuffle=False)
correct_forget_loader = loader.run('forget')
eval_forget_loader = loader.run('forget', shuffle=False)
train_labels=[]
for _,l,_ in train_loader:
    train_labels.extend(l.numpy())
#各クラスのサンプル数の出力
print(Counter(train_labels))
#格納されてるサプル数数の出力
print(f'forget sample num:{len(correct_forget_loader.dataset)}')
print(f'retain sample num:{len(correct_retain_loader.dataset)}')
print(f'test sanple num:{len(test_loader.dataset)}')
if args.noise_mode=='SDN':
    args.num_classes = 20 

with torch.no_grad():
    local_encoder = []
    local_cluster_labels= [] 
    for id, (inputs, targets, _) in enumerate(correct_retain_loader):
        inputs, targets=inputs.cuda(), targets.cuda()
        output, encoder=model(inputs, mode='t-SNE')
        local_cluster_labels.append(torch.softmax(output, dim=1))
        local_encoder.append(F.normalize(encoder, dim=1)) 
centers = mean(args, local_cluster_labels, local_encoder, correct_retain_loader)
mode='encoder'
feature_vector(args, model, eval_forget_loader, epoch=15, data='forget', modes=mode, centers=centers)
