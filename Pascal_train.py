# -*- coding: utf-8 -*-

import os, sys, numpy as np
import argparse

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PascalLoader import DataLoader
import torch.nn as nn
from helper import AverageMeter
from sklearn.metrics import average_precision_score
#sys.path.append('/home/maple/Pascal/model/')
from collections import OrderedDict
import deepcoder_classification_msssim_depthwise128_reduce_pool_v2 as JM
import torch_msssim
import math
import torchvision
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train network on Pascal VOC 2007')
parser.add_argument('-pascal_train_path', default='/data1/VOC2012/', help='Path to Pascal VOC 2007 folder')
parser.add_argument('--epochs', default=201, type=int, help='gpu id')
parser.add_argument('--batch', default=32, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--crops', default=2, type=int, help='number of random crops during testing')
args = parser.parse_args()


def main():


    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # DataLoader initialize
    train_data = DataLoader(args.pascal_train_path, 'train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch,
                                               shuffle=True)
    val_data = DataLoader(args.pascal_train_path, 'val', transform=val_transform, random_crops=args.crops)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=1,
                                             shuffle=False)
    
    
    model = JM.CodedVision(128, 128).cuda()
    #model.classifier.fc = nn.Linear(1024, 102)
    pretrained_dict = torch.load('./deepcoder_msssim_c128-128_v4pool_mid_bpp_0.6552_MSSSIM_0.9891_state_dict.pth')
    model_dict = model.state_dict()
    
    #new_state_dict = OrderedDict()
    #for k, v in pretrained_dict.items():
        #name = k[7:]
        #new_state_dict[name] = v
    
    new_state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #new_state_dict = {k:v for k,v in new_state_dict.items() if k in model_dict}
    model_dict.update(new_state_dict)    
    model.load_state_dict(model_dict)
    model.classifier.fc = nn.Linear(1024,21)
    model.cuda()
    
    
    for param in model.Encoder.parameters():
        param.requires_grad = False
    for param in model.Decoder.parameters():
        param.requires_grad = False
    for param in model.Hypercoder.parameters():
        param.requires_grad = False
    for param in model.estimator.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    init_lr = args.lr	
    optimizer = torch.optim.Adam(model.classifier.fc.parameters(), lr=args.lr,betas = (0.9,0.99))
    #optimizer = torch.optim.Adam([{'params':net.classifier.parameters()}], lr=args.lr)
    '''
    optimizer = torch.optim.Adam([{'params':net.Encoder.parameters(),'lr':args.lr},
                                  {'params':net.Decoder.parameters(),'lr':args.lr},
                                  {'params':net.Hypercoder.parameters(),'lr':args.lr},
                                  {'params':net.estimator.parameters(),'lr':args.lr},
                                  {'params':net.classifier.parameters(),'lr':args.lr},
                                  ]                                  
                                  ,lr=args.lr                                 
                                 )
    '''
    #optimizer = torch.optim.Adam([{'params':net.Decoder.parameters()}],lr=args.lr_codec)
    #optimizer = torch.optim.Adam([{'params':net.parameters()}],lr=args.lr)
    #criterion1 = nn.MSELoss().cuda()
    criterion2 = nn.MultiLabelSoftMarginLoss().cuda()
    criterion3 = torch_msssim.MS_SSIM(max_val=1).cuda()
    #entropy_function = rloss.rate_loss().cuda()
    ############## TRAINING ###############
    print('Start training: lr %f, batch size %d' % (args.lr, args.batch))

    # Train the Model
    #bpp_ae = 0.0
    #bpp_hyper = 0.0
    #mAP, MSE = test(net,val_loader,args,criterion1)
    
    for epoch in range(args.epochs):
        Loss = AverageMeter('Loss', ':.4f')
        Loss_cls = AverageMeter('Loss_cls', ':.4f')
        Loss_rec = AverageMeter('Loss_rec', ':.5f')
        Bpp_ae = AverageMeter('bpp', ':.3f')
        Bpp_hyper = AverageMeter('Bpp-hyper', ':.3f')
        running_loss = 0.0
        if epoch==20:
            #args.lr = 0.5*(1+math.cos(epoch*math.pi/args.epochs))*init_lr
            args.lr = args.lr/10
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        if epoch==40:
            #args.lr = 0.5*(1+math.cos(epoch*math.pi/args.epochs))*init_lr
            args.lr = args.lr/10
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            #print(labels.size())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            
            _,bpp,cls_output = model(images,Training=True)
            bpp = torch.mean(bpp)

            #loss_MSSSIM = criterion3(output, images)  
            loss_cls = criterion2(cls_output, labels) ###phase1, use the pretrained compression model and train the classifier branch for classification only
            #loss_rec = 1 - criterion3(output,images)
            #loss = 20*loss_cls + 0.1*loss_rate + 20*loss_rec ###phase2, when classification near to converge, try joint training

            loss = loss_cls
            loss.backward()
            optimizer.step()
            Loss.update(loss.item(), images.size(0))
            Loss_cls.update(loss_cls.item(),images.size(0))
            #Loss_rec.update(loss_rec.item(),images.size(0))
            #Bpp_ae.update(train_bpp_ae.item(), images.size(0))
            #Bpp_hyper.update(train_bpp_hyper.item(), images.size(0))
            '''
            if (i) % 10 == 0:
                print(
                'Epoch:[{0}][{1}/{2}]\t'
                'Loss:{losses.val:.4f}({losses.avg:.4f})\t'
                'Loss_cls:{losses_cls.val:.4f}({losses_cls.avg:.4f})\t'
                'Loss_rec:{losses_rec.val:.4f}({losses_rec.avg:.4f})\t'
                'Loss_ae:{losses_ae.val:.4f}({losses_ae.avg:.4f})\t'
                'Loss_hyper:{losses_hyper.val:.4f}({losses_hyper.avg:.4f})\t'
                .format(epoch,i, len(train_loader), losses=Loss,losses_cls = Loss_cls,losses_rec=Loss_rec,losses_ae=Bpp_ae,losses_hyper=Bpp_hyper)               
                  )
            '''
            if (i) % 10 == 0:
                print(
                'Epoch:[{0}][{1}/{2}]\t'
                'Loss:{losses.val:.4f}({losses.avg:.4f})\t'
                'Loss_cls:{losses_cls.val:.4f}({losses_cls.avg:.4f})\t'
                .format(epoch,i, len(train_loader), losses=Loss,losses_cls = Loss_cls)               
                  )
        if epoch % 10 == 0:
            mAP,MSSSIM = test(model,val_loader,criterion3,args)
            #bpp = Bpp_ae.avg+Bpp_hyper.avg
            #print('bpp:',bpp)
            model = model.train()
            torch.save(model, './Pascal_2c_epoch_' + str(epoch) + '_mAP_'+str(mAP)+'_MSSSIM_'+str(1-MSSSIM)+'_rate_'+str(bpp) +'.pkl')

def test(model, val_loader,criterion1,args):
    mAP = []
    MSSSIM = []
    model = model.cuda()
    model = model.eval()
    print('>>Testing...')
    for i, (images, labels) in enumerate(val_loader):

        #print(images.shape)
        images = images.view((-1, 3, 256, 256))
        
        images = images.cuda()
        labels = labels.cuda()
               
        B,C,H_ORG, W_ORG= images.shape
        # divided by 8
        H_PAD = int(8.0 * np.ceil(H_ORG / 8.0))
        W_PAD = int(8.0 * np.ceil(W_ORG / 8.0))
        im = torch.zeros([B, C, H_PAD, W_PAD], dtype=torch.float).cuda()
        im[:, :, :H_ORG, :W_ORG] = images
        
        output,bpp,cls_output = model(images,Training=False)
        
        #encoded_hyper = model.Hypercoder.encoder(encoded_raw, training=True)
        #print('encoded_hyper:',encoded_hyper.shape)
        #loc, scale = model.Hypercoder.decoder(encoded_hyper)
        #cls_out = model(im)
        outputs = cls_output.data.cpu()
        outputs = outputs.view((-1, args.crops, 21))
        outputs = outputs.mean(dim=1).view((-1, 21))
        mAP.append(compute_mAP(labels, outputs))
        
        msssim = 1 - criterion1(output,images)
        quality = msssim.data.cpu().numpy()
        MSSSIM.append(quality)
        
    print('finish test')
    print('TESTING:  mAP %.2f%%' % (100 * np.mean(mAP)))
    print('TESTING:  MSSSIM %.4f%%' % (np.mean(MSSSIM)))
    return np.mean(mAP),np.mean(MSSSIM)

def compute_mAP(labels, outputs):
    y_true = labels.cpu()
    y_pred = outputs.cpu()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)
				
if __name__ == "__main__":
    main()
