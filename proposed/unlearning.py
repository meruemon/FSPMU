
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
    
import copy
import torch.nn as nn
from torch.autograd import Variable
from typing import List
import itertools
from tqdm.autonotebook import tqdm
from models import *
import models
from logger import *
# from main import warmup_pro, test
import wandb
import sys
from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss

from thirdparty.repdistiller.helper.loops import train_forget, train_distill,  validate
from thirdparty.repdistiller.helper.pretrain import init
import warnings

# 警告を非表示にする
warnings.filterwarnings("ignore", category=UserWarning)

cls_non_avearge = nn.CrossEntropyLoss(reduction='none')

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


from utils import *

def l2_penalty(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss
def write_files(f_name, write_array, epoch, net=1):
    with open(f_name, "a") as f:
        f.write("Epoch {} (Net {}):\n".format(epoch, net))
        np.savetxt(f, write_array.reshape(1, -1), 
                   fmt='%0.4f', delimiter=' ', newline='\n')

def parameter_count(model):
    count=0
    for p in model.parameters():
        count+=np.prod(np.array(list(p.shape)))
    print(f'Total Number of Parameters: {count}')


def re_create_loader(args, loader):
    eval_train_loader = loader.run('noise_pred', shuffle=False)
    if args.pred == 'k-means' or args.pred =='dbscan':
        if 'resnet' in args.model:
            feat_dim=512
        else: feat_dim=192
        encoders = []
        labels = []
        c_or_n_s = []
        with torch.no_grad():
            for id, (input, noisy, clean, c_or_n) in enumerate(eval_train_loader):
                inputs, targets = input.cuda(), noisy.cuda()
                output, encoder = model_s(inputs, mode='t-SNE')
                # print(id)
                encoder = encoder.cpu() 
                encoders.append(F.normalize(encoder, dim=1)) 
                targets = targets.cpu()
                labels.extend(targets)
                c_or_n_s.extend(c_or_n)
        encoders = np.vstack(encoders)
        sfeature_vector_tsne = TSNE(perplexity=50, n_components=2, learning_rate=350, random_state=args.seed).fit_transform(encoders)
        labels = np.array(labels)
        if args.pred == 'k-means':
            kmeans = KMeans(n_clusters=40, max_iter=30, init="random")
            cluster = kmeans.fit_predict(sfeature_vector_tsne)
        elif args.pred == 'dbscan':
            db_scan = DBSCAN(eps=args.eps, min_samples=args.min_sample)
            cluster = db_scan.fit_predict(encoders)
            print(f'eps:{args.eps}\tmin_sample:{args.min_sample}\n')
        print(cluster)
        unipue_cluster = np.unique(cluster)
        print(len(set(cluster)))
        # exit()

    else:
        losses=[]
        c_or_n_s = []
        with torch.no_grad():
            for batch_idx, (data, target, _, c_or_n) in enumerate(eval_train_loader):
                data, target = data.cuda(), target.cuda()
                out = model_s(data)
                loss =cls_non_avearge(out, target)
                losses.extend(loss)
                c_or_n_s.extend(c_or_n)
        losses = np.array([loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in losses])
        print(f'clean sample num:{sum(c_or_n_s)}')
        if args.pretrain_method!=None:
            first_losses=[]
            with torch.no_grad():
                for batch_idx, (data, target, _, c_or_n) in enumerate(eval_train_loader):
                    data, target = data.cuda(), target.cuda()
                    out = model_firstgmm(data)
                    loss =cls_non_avearge(out, target)
                    first_losses.extend(loss)
            first_losses = np.array([loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in first_losses])
            first_losses = first_losses.reshape(-1, 1)
            gmm_first=GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm_first.fit(first_losses)
            prob_first = gmm_first.predict_proba(first_losses)
            prob_first = prob_first[:, gmm_first.means_.argmin()]
            pred_first = (prob_first > args.gmm_threshhold)
            pred_ids_first = np.where(pred_first==False)[0]


        if args.pred == 'gmm':
            losses = losses.reshape(-1, 1)
            gmm=GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(losses)
            prob = gmm.predict_proba(losses)
            prob = prob[:, gmm.means_.argmin()]
            pred = (prob > args.gmm_threshhold)
            pred_ids = np.where(pred==True)[0]
            print(f'pred clean num:{sum(pred)}')
            print(sum(np.array(c_or_n_s)[np.array(pred)]))

            if args.pretrain_method!=None:
                # use prob_file
                # with open(prob_file, 'r') as f:
                    # for id, line in enumerate(f):
                    #     if id == 1:
                    #         prob_30epoch = np.array([float(i) for i in line.split(' ')])
                    #     if id == 2*300+1:
                    #         prob_300epoch = np.array([float(i) for i in line.split(' ')])
                    # first_pred_ids = np.where(np.array(prob_30epoch>args.gmm_threshhold)==False)[0]
                    # last_pred_ids = np.where(np.array(prob_300epoch>args.gmm_threshhold)==True)[0]

                print(len(pred_ids))
                common_ids = list(set(pred_ids_first) & set(pred_ids))
                print(f'common ids:{len(common_ids)}')
                    
                noisy_sample_id = np.where(pred==False)[0]   
                forget_setids = list(set(noisy_sample_id) & set(common_ids))
                common_ids = np.array(common_ids)
                forget_setids = np.array(forget_setids)
                print(f'forget set ids:{len(forget_setids)}')
                common_ids = np.array([False if i in common_ids else True for i in range(len(pred))])
                # exit()

    print(f'clean sample num:{len(pred.nonzero()[0])}')
    #recall, precision等を算出
    cm = confusion_matrix(c_or_n_s, pred)
    print(f'recall:{cm[1,1]/(cm[1,1]+cm[1,0])}')
    print(f'precision:{cm[1,1]/(cm[1,1]+cm[0,1])}')
    report = classification_report(c_or_n_s, pred)
    print(cm)
    print(report)
    if args.random:
        pred = random.choices([True, False], weights=[1-args.sample_rate, args.sample_rate], k=len(pred))
        pred = np.array(pred)

    re_retain_loader = loader.run('re-retain', pred=pred)
    if args.pretrain_method!=None: re_forget_loader = loader.run('re-forget', pred=common_ids)
    else: re_forget_loader = loader.run('re-forget', pred=pred)
    print(f'retain:{len(re_retain_loader.dataset)}\tforget:{len(re_forget_loader.dataset)}')
    if args.file_name!=None:
        with open(args.file_name, 'a') as f:
            f.write(f'{args.model}_{args.dataset}_{args.noise_mode}_{args.noise_rate}_{args.method}')
            f.write(f'recall:{cm[1,1]/(cm[1,1]+cm[1,0])}\tprecision:{cm[1,1]/(cm[1,1]+cm[0,1])}\n confusion matrix:\n{cm}\n reprt:\n{report}\n retain:{len(re_retain_loader.dataset)}\tforget:{len(re_forget_loader.dataset)}')
    
    return re_retain_loader, re_forget_loader
    
def scrub(args, e_1, e_2, centers=None):
    #pred noise
    if args.pred != '':
        retain_loader, forget_loader = re_create_loader(args, loader)
    elif args.sample_rate!=None:
        retain_loader, forget_loader = randam_retain_loader, randam_forget_loader
    #予測方法を指定していない場合はforgetにすべてのノイジーサンプルを分類された正しいものを用いる
    else:
        retain_loader, forget_loader = correct_retain_loader, correct_forget_loader

    #事前学習時の重心の計算
    with torch.no_grad():
        local_encoder = []
        local_cluster_labels= [] 
        for id, (inputs, targets, _) in enumerate(retain_loader):
            inputs, targets=inputs.cuda(), targets.cuda()
            output, encoder=model(inputs, mode='t-SNE')
            local_cluster_labels.append(torch.softmax(output, dim=1))
            local_encoder.append(F.normalize(encoder, dim=1)) 
    centers = mean(args, local_cluster_labels, local_encoder, retain_loader)


    #学習の開始
    for epoch in range(1, e_2 + 1):
        lr = sgda_adjust_learning_rate(epoch, args, optimizer)
        print("==> SCRUB unlearning ...")
        #事前学習済みデルルの精度計算
        print(f'epoch:{epoch}-学習前')
        acc_test = test(epoch, model_s, test_loader)
        acc_tests.append(100-acc_test)
        
        maximize_loss = 0
        if epoch <= e_1:
            #忘却
            maximize_loss = train_distill(epoch, forget_loader, module_list, None, criterion_list, optimizer, args, "maximize", centers=centers)
            print(f'epoch:{epoch}-forget後')
            acc_test = test(epoch, model_s, test_loader)
            f.write(f'\t\t\tafter_forget_acc_test:{acc_test}\n\n')
        #クラス構造の強化
        train_acc, train_loss = train_distill(epoch, retain_loader, module_list, None, criterion_list, optimizer, args, "minimize", centers=centers)

        print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
        f.write("epoch: {}\nmaximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}\n".format(epoch, maximize_loss, train_loss, train_acc))
        
        if epoch ==e_2:
            if args.save:
                os.makedirs(f'{args.dir_name}/net/', exist_ok=True)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': create_model,
                    'state_dict': model_s.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'seed':args.seed,
                }, filename='{}net/weight_save_{}.tar'.format(args.dir_name, epoch))
            if args.tsne!=0:
                feature_vector(args, model_s, eval_forget_loader, epoch=epoch, data='forget', modes=mode, centers=centers)
        

parser=argparse.ArgumentParser()
parser.add_argument('--model', default='preactresnet18')
parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--num_to_forget', type=int, default=None,
                        help='Number of samples of class to forget')
parser.add_argument('--noise_rate', type=float, default=None)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
parser.add_argument('--lossfn', type=str, default='ce',
                        help='Cross Entropy: ce or mse')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--print_freq', type=int, default=500, help='print frequency')

#損失項のhyper-parameter
parser.add_argument('--gamma', type=float, default=1, help='cross entropy in minimize')
parser.add_argument('--beta', type=float, default=0, help='cross entropy in minimize')
parser.add_argument('--alpha', type=float, default=0, help='cross entropy in minimize')
parser.add_argument('--delta', type=float, default=500, help='cos_sim in minimize')
parser.add_argument('--zeta', type=float, default=20, help='cos_sim in maximize')
parser.add_argument('--eta', type=float, default=1, help='loss_div in maximize')

parser.add_argument('--epochs', type=int, default=200, help='learning epoch of pre-train model')
parser.add_argument('--noise_mode', type=str, default='asym', choices=['sym', 'asym', 'SDN'], help='asym or sym or SDN(Subclass Domain Noise)')
parser.add_argument('--method', type=str, default='scrub')
parser.add_argument('--e_n', nargs='+', type=int, default=[5])
parser.add_argument('--e_r', nargs='*', type=int, default=[15])
parser.add_argument('--tsne', type=int, default=0, help='0:none tsne, 1:after e_n and e_r tsne, 2:after e_n and e_r andb est model tsne')
parser.add_argument('--save', type=bool, default=False, help='model_weight save')
parser.add_argument('--gpu', default=0, type=int, help='use gpu-id')
parser.add_argument('--forget_bs', type=int, default=512)
parser.add_argument('--retain_bs', type=int, default=128)
parser.add_argument('--kd_T', type=float, default=0.5, help='kd_loss parameter (loss * args.kd_T**)')
parser.add_argument('--pred', type=str, default='', choices=['gmm', 'cls','k-means'], help='how to predict noise-sample')
parser.add_argument('--cls_threshhold', type=float, default=10.)
parser.add_argument('--gmm_threshhold', type=float, default=0.5)
#parameter of t-SNE
parser.add_argument('--tsne_lr', type=int, default=350)
parser.add_argument('--tsne_per', type=int, default=30)
#parameter of db-scan
parser.add_argument('--eps', type=float, default=0.02)
parser.add_argument('--min_sample', type=int, default=100)
parser.add_argument('--file_name', type=str, default=None)
parser.add_argument('--random', default=False)
parser.add_argument('--pretrain_method', default=None, choices=[None, 'DivideMix', 'ProMix', 'LongRe-Mix'])
parser.add_argument('--sample_rate', type=float, default=None)
parser.add_argument('--scratch', action='store_true', default=False)

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

class_to_forget = args.forget_class
num_classes=args.num_classes
seed = args.seed


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

#loaderの生成
if args.pretrain_method==None: noise_file = f'../../weight/unlearning/{args.model}_{args.dataset}_{args.noise_rate}_{args.noise_mode}/net_seed{args.seed}/{args.noise_rate:.2f}_{args.noise_mode}.json'
elif args.pretrain_method=='DivideMix':  noise_file = f'../../weight/{args.pretrain_method}/seed_{args.seed}/{args.dataset}_{args.noise_rate}_{args.noise_mode}/noise_file.json'
elif args.pretrain_method=='ProMix':  noise_file = f'../../weight/{args.pretrain_method}/{args.dataset}_{args.noise_rate}_{args.noise_mode}/noise_file.json'
elif args.pretrain_method=='LongReMix': noise_file = '../../weight/LongReMix/{}/{:.2f}_{}.json'.format(args.dataset, args.noise_rate, args.noise_mode)

print(f'noise file name:\n{noise_file}')
loader = datasets.cifar_dataloader(args.dataset,r=args.noise_rate,noise_mode=args.noise_mode,
                                     batch_size=args.batch_size,num_workers=12,\
    root_dir=args.dataroot,noise_file=noise_file,\
        retain_bs=args.retain_bs, forget_bs=args.forget_bs, sample_rate=args.sample_rate)
from collections import Counter
train_loader = loader.run('eval_train')
test_loader = loader.run('test')

file_name=f'{args.model}_{args.dataset}_{args.noise_rate}_{args.noise_mode}'
torch.cuda.set_device(args.gpu)
def create_model():
    num_classes = 20 if args.noise_mode=='SDN' and args.dataset=='cifar100' else args.num_classes
    model = models.get_model(args.model, num_classes=num_classes)
    model = model.cuda()
    return model
if args.pretrain_method==None: 
    checkpoint = torch.load('../../weight/unlearning/{}/net_seed{}/net/weight_save_{:04d}.tar'.format(file_name, args.seed, args.epochs), map_location=f'cuda:{args.gpu}')
    model = create_model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    model.load_state_dict(checkpoint['state_dict'])
elif args.pretrain_method=='DivideMix': 
    checkpoint = torch.load('../../weight/%s/seed_%d/%s_%.1f_%s/net1.pth.tar'%(args.pretrain_method, args.seed, args.dataset,args.noise_rate,args.noise_mode), map_location=f'cuda:{args.gpu}')
    prob_file = '../../weight/%s/seed_%d/%s_%.1f_%s/prob.txt'%(args.pretrain_method, args.seed, args.dataset,args.noise_rate,args.noise_mode)
    model = create_model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    model.load_state_dict(checkpoint['state_dict'])
elif args.pretrain_method=='ProMix': 
    args.model += '_promix'
    checkpoint1 = torch.load('../../weight/%s/seed_%d/%s_%.1f_%s/net1.pth.tar'%(args.pretrain_method, args.seed, args.dataset,args.noise_rate,args.noise_mode), map_location=f'cuda:{args.gpu}')
    checkpoint2 = torch.load('../../weight/%s/seed_%d/%s_%.1f_%s/net2.pth.tar'%(args.pretrain_method, args.seed, args.dataset,args.noise_rate,args.noise_mode), map_location=f'cuda:{args.gpu}')
    model1 = create_model()
    model2 = create_model()
    model1.load_state_dict(checkpoint1['state_dict'])
    model2.load_state_dict(checkpoint2['state_dict'])
    acc1 = test(0, model1, test_loader)
    acc2 = test(0, model2, test_loader)
    if acc1>acc2:
        cheackpoint=checkpoint1.copy()
    else:
        checkpoint=checkpoint2.copy()
    model = create_model()
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    prob_file = '../../weight/%s/seed_%d/%s_%.1f_%s/prob_file.txt'%(args.pretrain_method, args.seed, args.dataset,args.noise_rate,args.noise_mode)
    del model1, model2, checkpoint1, checkpoint2
elif args.pretrain_method=='LongRe-Mix': 
    noise_rate = float(str(args.noise_rate).replace('_','.'))
    print('../LongReMix-main/hcs/hcs_{}_{:.2f}_{}_cn5_run0.pth.tar'.format(args.dataset, noise_rate,args.noise_mode))
    # checkpoint = torch.load('../LongReMix-main/hcs/hcs_{}_{:.2f}_{}_cn5_run0.pth.tar'.format(args.dataset, noise_rate,args.noise_mode))
    checkpoint = torch.load('../../checkpoint-server/{}_{:.2f}_{}/model_ckpt.pth.tar'.format(args.dataset, noise_rate, args.noise_mode), map_location=f'cuda:{args.gpu}')
    model1 = create_model()
    model2 = create_model()
    model1.load_state_dict(checkpoint['state_dict1'])
    model2.load_state_dict(checkpoint['state_dict2'])
    acc1 = test(0, model1, test_loader)
    acc2 = test(0, model2, test_loader)
    model = create_model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    if acc1>acc2:
        model.load_state_dict(checkpoint['state_dict1'])
        optimizer.load_state_dict(checkpoint['optimizer1'])
    else:
        model.load_state_dict(checkpoint['state_dict2'])
        optimizer.load_state_dict(checkpoint['optimizer2'])
    prob_file = ''
    del model1, model2, checkpoint


#modelの設定
model.cuda()
parameter_count(copy.deepcopy(model))
for p in model.parameters():
    p.data0 = p.data.clone()

t1 = time.time()
model_firstgmm = create_model()
# 学習初期のGMMのためのモデル学習
if args.pretrain_method!=None and args.pred=='gmm' and not args.scratch:
    
    optimizer_firstgmm = optim.SGD(model_firstgmm.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    if args.dataset=='cifar10': epochs_ = 1
    else: epochs_ = 30
    for epoch in range(1,  epochs_+ 1):
        lr=args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr            

        warmup(epoch,model_firstgmm,optimizer_firstgmm,train_loader, args)  

        acc = test(epoch,model_firstgmm,test_loader)

correct_retain_loader = loader.run('retain')
eval_retain_loader = loader.run('retain', shuffle=False)
correct_forget_loader = loader.run('forget')
eval_forget_loader = loader.run('forget', shuffle=False)
if args.sample_rate!=None:
    randam_forget_loader, randam_retain_loader = loader.run('rand-sampling', shuffle=True)
#格納されてるサプル数数の出力
print(f'forget sample num:{len(correct_forget_loader.dataset)}')
print(f'retain sample num:{len(correct_retain_loader.dataset)}')
print(f'test sanple num:{len(test_loader.dataset)}')
if args.noise_mode=='SDN':
    args.num_classes = 20 

#scrubで設定されてたハイパーパラメータ
args.optim = 'sgd'
args.smoothing = 0.5
args.clip = 0.5
args.sstart = 10
args.distill = 'kd'
args.sgda_learning_rate = 0.0005
args.lr_decay_epochs = [7,10,10]
args.lr_decay_rate = 0.1
args.sgda_weight_decay = 5e-4
args.sgda_momentum = 0.9
#結果格納用リスト
acc_rs = [0]*15
acc_fs = [0]*15
acc_f_cs=[0]*15
acc_vs = [0]*15
acc_tests=[]
mode='encoder'

#モデル・損失のリストの作成
model_t = copy.deepcopy(model)  #teacher model
model_s = copy.deepcopy(model)  #student model
module_list = nn.ModuleList([])
module_list.append(model_s)
trainable_list = nn.ModuleList([])
trainable_list.append(model_s)
criterion_cls = nn.CrossEntropyLoss()
criterion_div = DistillKL(args.kd_T)
criterion_kd = DistillKL(args.kd_T)
criterion_list = nn.ModuleList([])
criterion_list.append(criterion_cls)    # classification loss
criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
criterion_list.append(criterion_kd)     # other knowledge distillation loss
# optimizer
if args.optim == "sgd":
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=args.sgda_learning_rate,
                          momentum=args.sgda_momentum,
                          weight_decay=args.sgda_weight_decay)
elif args.optim == "adam": 
    optimizer = optim.Adam(trainable_list.parameters(),
                          lr=args.sgda_learning_rate,
                          weight_decay=args.sgda_weight_decay)
# elif args.optim == "rmsp":
#     optimizer = optim.RMSprop(trainable_list.parameters(),
#                           lr=args.sgda_learning_rate,
#                           momentum=args.sgda_momentum,
#                           weight_decay=args.sgda_weight_decay)
module_list.append(model_t)

if torch.cuda.is_available():
    module_list.cuda()
    criterion_list.cuda()
    import torch.backends.cudnn as cudnn


#結果保存先filepathとディレクトリの作成
from datetime import datetime
import pytz
now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%H_%M")
noise_rate_name=str(args.noise_rate).replace('.', '_')
if args.pretrain_method==None: dir_name='../../result/unlearning/pred_{}/{}/{}/{}_{}/{}_seed_{}/{}/'.\
    format(args.pred, args.method, args.dataset, args.noise_mode, noise_rate_name, args.model, args.seed, date_time)
else: dir_name='../../result/unlearning/{}/pred_{}/{}/{}/{}_{}/{}_seed_{}/{}/'.\
    format(args.pretrain_method, args.pred, args.method, args.dataset, args.noise_mode, noise_rate_name, args.model, args.seed, date_time)

os.makedirs(dir_name, exist_ok=True)
args.dir_name=dir_name
file_name=dir_name+'score.txt'

print('+'*50)
#学習の開始
if not args.scratch:
    with open(file_name, 'w') as f:
        args.file=f
        for e_1, e_2 in zip(args.e_n, args.e_r):
            scrub(args, e_1, e_2)
        del args.file     

    t2=time.time()
    print(f'training_time:{t2-t1}')
    acc_test = test(0, model_s, test_loader)
    acc_tests.append(100-acc_test)

    result = 'test_acc:\n original={:.2f}\t propose(last)={:.2f} propose(best)={:.2f}\ntraining time={:.2f}%'.format(100 - acc_tests[-1], 100-acc_tests[-1], 100 - np.min(acc_tests[1:]), t2 - t1)
    parameter=f'model:{args.model}, ,dataset:{args.dataset}, method:{args.method}, kd_T:{args.kd_T}, f_bs:{args.forget_bs}, r_bs:{args.retain_bs},\
        alpha:{args.alpha}, beta:{args.beta}, gamma:{args.gamma}, delta:{args.delta}, zeta:{args.zeta}, eta:{args.eta}'

    with open(file_name, 'a') as f:
        f.write(f'\n{args.noise_mode}.{args.noise_rate}:{parameter}\n')
        f.write(result)
    #configファイル(argsの保存ファイル)の作成
    args.device=str(args.device)
    with open("%s/config.json"%(dir_name), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    best_index=np.argmin(acc_tests[1:])
    print(f'\n{args.noise_mode}.{args.noise_rate}:{parameter}:best_index{best_index}\n{result}\n')
    with open('score.txt', 'a') as f:
        f.write(f'\n{args.noise_mode}.{args.noise_rate}:{parameter}:best_index{best_index}\n{result}\n')


    if args.file_name!=None:
        with open(args.file_name, 'a') as f:
            f.write(f'\n{args.noise_mode}.{args.noise_rate}:{parameter}:best_index{best_index}\n{result}\n')

else:
    print('Scratch learning start') 
    args.lr = 0.01
    args.epochs = 200
    args.step_size = args.epochs//2
    model_scratch = create_model()
    scratch_loader, _ = re_create_loader(args, loader)
    optimizer = torch.optim.SGD(model_scratch.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    accs = []
    for epoch in range(args.epochs+1):   
        lr=args.lr
        if epoch == args.step_size:
            lr /= 10      
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr            

        warmup(epoch,model_scratch,optimizer,scratch_loader, args)  


        acc = test(epoch,model_scratch,test_loader)
        accs.append(acc)
    t2 = time.time()
    print(f'training_time:{t2-t1}')
    if args.file_name!=None:
        with open(args.file_name, 'a') as f:
            parameter=f'model:{args.model}, ,dataset:{args.dataset}, method:{args.method}, kd_T:{args.kd_T}, f_bs:{args.forget_bs}, r_bs:{args.retain_bs},\
        alpha:{args.alpha}, beta:{args.beta}, gamma:{args.gamma}, delta:{args.delta}, zeta:{args.zeta}, eta:{args.eta}'
            f.write(f'\n{args.noise_mode}.{args.noise_rate}:{parameter}\n')
            f.write(f'acc_best:{np.max(accs)}\t acc_last:{accs[-1]}\t training_time:{t2-t1}\n')
