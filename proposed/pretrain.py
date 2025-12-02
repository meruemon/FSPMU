import argparse
import os
import copy
import models
import random
import sys
from collections import defaultdict
import datasets_multiclass as datasets
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import *
from logger import Logger
import shutil
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import warnings

# 警告を非表示にする
warnings.filterwarnings("ignore", category=UserWarning)

#++++++++++++++++++++++++++++++++++++++++aaaaaaa
# from divide
# test用関数
def test(epoch,net, data_loader, mode='Test'):
    # stick_seed()
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| %s Epoch #%d\t Accuracy: %.2f%%" %(mode, epoch,acc))  
    return acc
# 重み保存のため
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
# モデル定義
def create_model():
    num_classes = 20 if args.noise_mode=='SDN' and args.dataset=='cifar100' else args.num_classes
    model = models.get_model(args.model, num_classes=num_classes)
    model = model.cuda()
    return model

# 損失関数で使う関数の用意
CEloss = nn.CrossEntropyLoss() 

def stick_seed():
    manual_seed(args.seed)
    seed=args.seed
    os.environ['PYTHONHASEED']=str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def warmup_pro(epoch, net, optimizer, dataloader, args):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    # 学習するためのループ
    for batch_idx, (inputs_x1, labels_cpu, _) in enumerate(dataloader):       
        inputs_x1 = inputs_x1.cuda()
        labels = labels_cpu.cuda() 
        optimizer.zero_grad()
        output =  net(inputs_x1) 
        loss = CEloss(output, labels) 
    
        loss.backward()  
        optimizer.step() 
        # 表示
        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d] Loss: %.4f'
            %(epoch, args.epochs, 
                batch_idx+1, num_iter, loss.item()))
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--epochs', type=int, default=200, metavar='N')
parser.add_argument('--filters', type=float, default=1.0,
                    help='Percentage of filters')
parser.add_argument('--lossfn', type=str, default='ce',
                    help='Cross Entropy: ce or mse')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--model', default='preactresnet18')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_to_forget', type=int, default=None,
                    help='Number of samples of class to forget')
parser.add_argument('--noise_rate', type=float, default=None)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--step-size', default=100, type=int, help='learning rate scheduler')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='M',
                    help='Weight decay (default: 0.0005)')
parser.add_argument('--noise_mode', type=str, default='asym', choices=['sym', 'asym', 'SDN'], help='asym or sym or SDN(Subclass Domain Noise)')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--save', type = bool, default=False)
parser.add_argument('--tsne', type = bool, default=False)
parser.add_argument('--tsne_lr', type=int, default=350)
parser.add_argument('--tsne_per', type=int, default=30)
args = parser.parse_args()
#ノイズ率，クラス数，データパスの定義
if args.num_to_forget!=None:
    args.noise_rate=float(args.num_to_forget/50000)
elif args.noise_rate!=None:
    args.num_to_forget=int(50000*args.noise_rate)
else:
    raise ValueError("both num_to_forget and noise_rate!")
if args.noise_mode=='SDN' and args.dataset!='cifar100':
    raise ValueError("SDN noise use only cifar100 dataset")
args.dataroot=None
if args.dataset == 'cifar100':
    args.dataroot = '../../data/cifar100/cifar-100-python'
    args.num_classes = 100
else:
    args.dataroot = '../../data/cifar10/cifar-10-batches-py'
    if 'cifar5' in args.dataset:
        args.num_classes = 5
    elif 'cifar10' in args.dataset:
        args.num_classes = 10
args.r = args.noise_rate
args.forget_class=list(range(args.num_classes))
if args.step_size==None:args.step_size=args.epochs+1
use_cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if use_cuda else "cpu")
noise_mode=args.noise_mode



#seed系の固定
manual_seed(args.seed)
seed=args.seed
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

#create loader
noise_file = f'../../weight/{args.model}_{args.dataset}_{args.noise_rate}_{args.noise_mode}/net_seed{args.seed}/{args.noise_rate:.2f}_{args.noise_mode}.json'
print(args.dataroot)
loader = datasets.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,
                                     batch_size=args.batch_size,num_workers=12,\
    root_dir=args.dataroot,noise_file=noise_file)
from collections import Counter
train_loader = loader.run('eval_train')
test_loader = loader.run('test')
retain_loader = loader.run('retain')
eval_retain_loader = loader.run('retain', shuffle=False)
forget_loader = loader.run('forget')
eval_forget_loader = loader.run('forget', shuffle=False)
print(retain_loader, forget_loader)
train_labels=[]
for _,l,_ in train_loader:
    train_labels.extend(l.numpy())
print(Counter(train_labels))
retain_labels=[]
for _, l, _ in retain_loader:
    retain_labels.extend(l.numpy())
forget_labels = []
for _, l, _ in forget_loader:
     forget_labels.extend(l.numpy())
# print(Counter(forget_labels))
print(f'forget sample num:{len(forget_labels)}')
# print(Counter(retain_labels))
print(f'retain sample num:{len(retain_labels)}')

torch.cuda.set_device(args.gpu)
#モデルとSGDの作成
num_classes = args.num_classes
print(f"Number of Classes: {num_classes}")
print('| Building net')
model= create_model()
parameters = model.parameters()
optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)

path_name=f'../../weight/'
args.dir_name=f'{args.model}_{args.dataset}_{args.r}_{noise_mode}/net_seed{args.seed}/'
os.makedirs(path_name+args.dir_name+'net/', exist_ok=True)
args.dir_name = path_name + args.dir_name

# bestの保存変数
best = 0
t1 = time.time()
# 学習のループ
for epoch in range(args.epochs+1):   
    lr=args.lr
    if epoch >= args.step_size:
        lr /= 10      
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr            

    warmup_pro(epoch,model,optimizer,train_loader, args)  


    acc = test(epoch, model,eval_retain_loader, mode='Retain')
    acc = test(epoch,model,test_loader)

    # 以下重み保存
    if args.save:
        if ((epoch == args.epochs)):  
            # save_checkpointは84
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': create_model,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'seed':args.seed,
            }, filename='{}/net/weight_save_{:04d}.tar'.format(args.dir_name, epoch))
            print(f'save_weigh_last:{args.dir_name}')
        if acc > best:
            best = acc
            save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': create_model,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'seed':args.seed,
                }, filename='{}/net/weight_save_9999.tar'.format(args.dir_name))
        print(f'save_weigh_best\nepoch:{epoch}\tacc:{best}\n')
t2 = time.time()
# pre-trainの特徴量可視化
if args.tsne:
    centers=None
    with torch.no_grad():
        local_encoder = []
        local_cluster_labels= [] 
        for id, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets=inputs.cuda(), targets.cuda()
            output, encoder=model(inputs, mode='t-SNE')
            local_cluster_labels.append(torch.softmax(output, dim=1))
            local_encoder.append(F.normalize(encoder, dim=1)) 

        centers = mean(args, local_cluster_labels, local_encoder, train_loader)
    feature_vector(args, model, train_loader, epoch=epoch, data='train', centers=centers)
    feature_vector(args, model, eval_retain_loader, eval_forget_loader, epoch=epoch, data='forget', centers=centers)
    feature_vector(args, model, test_loader, epoch=epoch, data='test')

with open(args.dir_name+ 'result.txt', 'w') as f:
    f.write(f'best acc:{best}\nlearning time:{t2-t1}')