from .Unet import UNetSeeInDark
from .NAFNet import NAFNet
import torch.nn as nn



def unet(in_channels, out_channels, **kwargs):
    return UNetSeeInDark(in_channels, out_channels, **kwargs)


def naf(in_channels, out_channels, **kwargs):
    return  NAFNet(in_channels, out_channels, **kwargs)

def naf2(in_channels, out_channels, **kwargs):
    return  NAFNet(in_channels, out_channels, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 4], dec_blk_nums=[2, 2, 2, 2], **kwargs)

def naf3(in_channels, out_channels, **kwargs): # para 1,137,492
    return  NAFNet(in_channels, out_channels, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1], **kwargs)

def naf4(in_channels, out_channels, **kwargs): # para 19,732 ('4.1608G', '18.9640K') 0.26617980003356934
    return  NAFNet(in_channels, out_channels, middle_blk_num=8, **kwargs)