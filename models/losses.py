import torch.nn as nn
import torch

class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)

class KL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def clip_preseve_grad(self, x, min, max):
        x_clipped = x.clamp(min, max)
        return  (x_clipped-x).detach() + x
        

    def forward(self, predict, gt):
        # threshold = # 0.1 # 0.02 # 50 / (16383-512)
        threshold = 0.2
        mask = (gt < threshold) | (predict < threshold)
        mask = ~mask
        loss_kl = (predict[mask] * (predict[mask]/gt[mask]).log() + gt[mask] - predict[mask]).sum()
        loss_l1 = (predict[~mask] - gt[~mask]).abs().sum()
        # loss_kl = loss_kl.nan_to_num(0)
        # mask = loss_kl.isnan()
        loss = (loss_kl+loss_l1)/torch.numel(predict)
        # loss = predict * (predict.clamp(1/16383)/gt.clamp(1/16383)).log() + gt - predict 
        # loss = (predict - gt).abs()
        return loss.mean() 
    
def init_loss(opt):
    loss_dic = {}

    print('[i] Pixel Loss: {}'.format(opt.loss))

    pixel_loss = ContentLoss()
    if opt.loss == 'l1':
        pixel_loss.initialize(nn.L1Loss())
    elif opt.loss == 'l2':
        pixel_loss.initialize(nn.MSELoss())
    elif opt.loss == 'kl':
        pixel_loss.initialize(KL_Loss())
    loss_dic['pixel'] = pixel_loss

    return loss_dic
