import torch
import argparse
from data_loader import data_loader
import deepcoder_pytorch_v4pooling as dp
from helper import AverageMeter
#import entropy_pytorch
import numpy as np
import torch_msssim
import warnings
warnings.filterwarnings("ignore")

parse = argparse.ArgumentParser(description='pytorch verison of single-model-dic ')
parse.add_argument('-data', default='/data1/imagenet/',metavar='DIR',help= 'path to dataset')
#parse.add_argument('-data', default='/data1/deepcoder/',metavar='DIR',help= 'path to dataset')
parse.add_argument('-num_workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
parse.add_argument('-lr_opt', default=1e-4, type=float, metavar='learning rate', help='initial learning rate')
parse.add_argument('-epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
parse.add_argument('-batch_size',default=1, type=int, metavar='bz', help='mini-batch size')
parse.add_argument('-print_freq',default=100, type=int, metavar='N', help='print frequency ')

def main():
    global args, N, M
    args = parse.parse_args()
    N = 128
    M = 128
    device = torch.cuda.current_device()
    print(device)
    model = dp.single_model_coder(N,M).cuda()
    #model_dict = torch.load('./model/deepcoder_lam=30_msssim_c128-128_epoch0_step_3_bpp_0.3226_MSSSIM_0.9716.pkl')
    #model.load_state_dict(model_dict)
    #model = torch.load('./model/deepcoder_lam=30_msssim_c128-128_epoch0_step_3_bpp_0.3226_MSSSIM_0.9716.pkl')

    
    train_loader, eval_loader = data_loader(root=args.data, batch_size=args.batch_size, workers=args.num_workers)
    #criterion =  torch_msssim.MS_SSIM(max_val=1).cuda()
    criterion =  torch.nn.MSELoss().cuda()
    optimizer_dp = torch.optim.Adam([{'params': model.Encoder.parameters()},{'params': model.Decoder.parameters()},{'params': model.Hypercoder.parameters()}, {'params': model.estimator.parameters()}], lr=args.lr_opt)

    for i in range(args.epochs):
        train(model,train_loader, criterion,  optimizer_dp, args.print_freq, i)
        if i==1:
            args.lr_opt = args.lr_opt/10
            for param_group in optimizer_dp.param_groups:
                param_group['lr'] = args.lr_opt
        

def train(model,train_loader,criterion, optimizer_dp, print_freq,epoch):
    Loss = AverageMeter()
    Bpp_ae = AverageMeter()
    Bpp_hyper = AverageMeter()
    Loss_msssim = AverageMeter()
    Loss_MSE = AverageMeter()

    lam = 500
    beta = 0.8
    bpp_ae = 0.0
    bpp_hyper = 0.0
    ls = 0.0

    for step, (img,target) in enumerate(train_loader):
        img = img

        #print(img.shape)
        B,C,H_ORG, W_ORG= img.shape
        #print(img.shape)
        if H_ORG>1024:
            H_ORG=1024
            img = img[:,:,:H_ORG,:]
        if W_ORG>1024:
            W_ORG=1024
            img = img[:,:,:,:W_ORG]
        # divided by 8
        H_PAD = int(64.0 * np.ceil(H_ORG / 64.0))
        W_PAD = int(64.0 * np.ceil(W_ORG / 64.0))
        im = torch.zeros([B, C, H_PAD, W_PAD], dtype=torch.float).cuda()
        im[:, :, :H_ORG, :W_ORG] = img

        num_pixels = im.size(0)*im.size(2)*im.size(3)
        encoded_raw, encoded = model.Encoder(im, training=True)
        output = model.Decoder(encoded)
        encoded_hyper = model.Hypercoder.encoder(encoded_raw, training=True)
        loc, scale = model.Hypercoder.decoder(encoded_hyper)
          
        upper = encoded + 0.5
        lower = encoded - 0.5
        sign = (upper + lower - loc).sign()
        upper = - sign * (upper - loc) + loc
        lower = - sign * (lower - loc) + loc

        upper = model.Hypercoder.cdf_hyper(upper, loc, scale)
        lower = model.Hypercoder.cdf_hyper(lower, loc, scale)

        p_laplace = (upper - lower).abs()
        p_laplace[p_laplace <= 1e-6] = 1e-6

        train_bpp_ae = (torch.sum(torch.log(p_laplace))) / -(torch.log(torch.Tensor([2.0]).cuda())) / num_pixels

        encoded_x = encoded_hyper.view(-1,1,N)
        encoded_x = encoded_x.permute(2, 1, 0)
        likelihoods = model.estimator(encoded_x)
        train_bpp_hyper = torch.sum(torch.log(likelihoods)) / -(torch.log(torch.Tensor([2.0]).cuda())) / num_pixels

        loss_MSE = criterion(output, im)
        #msssim = criterion(output,im)
        #loss_msssim = 1. - msssim
        loss = lam*(loss_MSE) + beta*(train_bpp_ae + train_bpp_hyper)
        optimizer_dp.zero_grad()
        loss.backward()
        optimizer_dp.step()

        Loss.update(loss.item(), im.size(0))
        #Loss_msssim.update(msssim.item(), im.size(0))
        Loss_MSE.update(loss_MSE.item(), im.size(0))
        Bpp_ae.update(train_bpp_ae.item(), im.size(0))
        Bpp_hyper.update(train_bpp_hyper.item(), im.size(0))

        if step % print_freq == 0:
            print(
                'Epoch:[{0}][{1}/{2}]\t'
                'Loss:{losses.val:.4f}({losses.avg:.4f})\t'
                'Loss_MSE:{loss_mse.val:.4f}({loss_mse.avg:.4f})\t'
                'Bpp_ae:{Bpp_ae.val:.4f}({Bpp_ae.avg:.4f})\t'
                'Bpp_hyper:{Bpp_hyper.val:.4f}({Bpp_hyper.avg:.4f})\t'.format(epoch,step, len(train_loader), losses=Loss,loss_mse=Loss_MSE,Bpp_ae=Bpp_ae,Bpp_hyper=Bpp_hyper)
                  )
        if step % 100000 == 0:
            bpp ="%.4f"%(Bpp_ae.avg + Bpp_hyper.avg)
            mse_print = "%.4f"%Loss_MSE.avg
            torch.save(model,'./model/deepcoder_lam=500_beta=0.8_mse_c128-128_epoch'+str(epoch)+'_step_'+str(int(step/100000))+'_bpp_'+str(bpp)+'_MSSSIM_'+str(mse_print) +'.pkl')
        if step/200000 == 3:
            args.lr_opt = args.lr_opt/10
            for param_group in optimizer_dp.param_groups:
                param_group['lr'] = args.lr_opt

def eval (deepcoder,hypercoder, eval_loader,criterion,print_freq):
    Loss = AverageMeter()
    for step ,img in range(eval_loader):
        img = img.cuda()
        output = model(img)
        loss = criterion(output, img)
        Loss.update(loss.item(),img.size(0))
        if step % print_freq :
            print('Epoch:[{0}/{1}]\t', 'Loss: {losses.val:.4f}({losses.avg:.4f})\t'.format(step,len(eval_loader),losses=Loss))

if __name__ == '__main__':
    main()
