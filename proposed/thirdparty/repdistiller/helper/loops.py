from __future__ import print_function, division

import sys
import time
import torch
from torch import nn
from itertools import cycle

from .util import AverageMeter, accuracy, param_dist



import numpy as np
import random
import os
import torch.nn.functional as F

cos_sim = nn.CosineSimilarity(dim=1).cuda() # cos類似度
cos_sim_with_all_centers=nn.CosineSimilarity(dim=2).cuda()

def train_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, opt, split, quiet=False, centers=None ,center=None):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()
    args=opt
    f=args.file
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        cos_loss=0
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, clean = data
                        
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
                index = index.cuda()

        # ===================forward=====================
        logit_s , features= model_s(input, mode='t-SNE')
        with torch.no_grad():
            logit_t = model_t(input)


        loss_cls = criterion_cls(logit_s, target)

        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        else:
            raise NotImplementedError(opt.distill)
        if args.method=='pro' or args.method=='pro-momentum':
            if split=='minimize':
                output, encoder=model_s(input, mode='t-SNE')
                
                #重心と特徴量の類似度を計算
                cos_loss=cos_sim(encoder, centers[target].detach()).mean()
                loss = opt.gamma * loss_cls  +\
                    opt.beta * loss_kd - opt.delta * cos_loss
                if random.randint(0, 50)==0 and args.dataset=='cifar10':
                    f.write(f'retain\tloss_cls:{loss_cls}, cos_loss:{cos_loss}\n')
                elif args.dataset=='small_cifar5':
                    f.write(f'retain\tloss_cls:{loss_cls}, loss_div:{loss_div}, cos_loss:{cos_loss}\n')
                    
            elif split=='maximize':
                cos_loss=0.
                _, encoder=model_s(input, mode='t-SNE')
                #各サンプルの特徴量とすべての重心との類似度を求めるために拡張してる
                encoder_classes_5 = encoder.unsqueeze(1).repeat(1, args.num_classes, 1)
                centers_5=centers.unsqueeze(0).repeat(len(target), 1, 1)
                cos_loss=cos_sim_with_all_centers(encoder_classes_5, centers_5)
                # cos_loss = (cos_loss+1)/2
                #それぞれの重心とどれだけ離れてるかによって重み付けしてる
                #近いほどlossを大きくしてる
                cos_T = 0.5
                cos_t = cos_loss**(1/cos_T) 
                cos_dis = (cos_t/cos_t.sum(dim=1, keepdim=True)).detach()
                cos_loss=cos_loss * cos_dis.detach()#+ random.uniform(-100., 100.)
                cos_loss = cos_loss.sum(dim=1)
                cos_loss=cos_loss.mean()
                #損失とそれぞれの重心とどれだけ離れてるかの出力
                if random.randint(0, 50)==0 and args.dataset=='cifar10':
                    f.write(f'forget\t;loss_div{loss_div}, cos_loss:{cos_loss}\n')
                    # for label, cos in zip(target, cos_dis):
                    #     f.write(f'target:{label}, clean:{clean}\n{cos}\n')
                elif args.dataset=='small_cifar5':
                    f.write(f'forget\t;loss_div{loss_div}, cos_loss:{cos_loss}\n')
                    # for label, c, cos in zip(target, clean, cos_dis):
                    #     f.write(f'target:{label}, clean:{c}\n{cos}\n')
            
                loss = -opt.eta*loss_div + opt.zeta * cos_loss

        elif args.method=='scrub':
            #print('in_scrub')
            if split == "minimize":
                loss = opt.gamma * loss_cls  + opt.beta * loss_kd
            elif split == "maximize":
                loss = -opt.eta * loss_div

        

        if split == "minimize" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1,1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            # if random.randint(0, 100)==0:print(loss, losses)
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))
        elif split == "linear" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            kd_losses.update(loss.item(), input.size(0))


        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(model_s.parameters(), 1.0)
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        if not quiet:
            if split == "minimize":
                if idx % opt.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1))
                    sys.stdout.flush()
    if split == "minimize":
        if not quiet:
            print(' * Acc@1 {top1.avg:.3f} '
                  .format(top1=top1))

        return top1.avg, losses.avg
    else:
        return kd_losses.avg


def validate(val_loader, model, criterion, opt, quiet=False, clean_label_val=False):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target, clean) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = clean.cuda() if clean_label_val else target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if not quiet:
                if idx % opt.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           idx, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5))
        if not quiet:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
