import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision.utils import save_image

class UNetSeeInDark(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, **kwargs):
        super(UNetSeeInDark, self).__init__()
        self.resid = kwargs.pop("resid", False)
        self.with_photon = kwargs.get('with_photon', False)
        self.concat_origin = kwargs.get('concat_origin', False)
        self.adaptive_res_and_x0 = kwargs.get('adaptive_res_and_x0', False)
        self.in_channels = in_channels
        times = 1
        if self.with_photon:
            times += 1
        if self.concat_origin:
            times += 1
            
        if self.resid: print("[i] predict noise instead of clean image")#logging.info("predict noise instead of clean image")
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels*times, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        
        if self.adaptive_res_and_x0:
            print("[i] Using adaptive_res_and_x0. ")
            out_channels = out_channels * 2 + 1
            self.out_channels = out_channels
        
        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)
        

            
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        
        conv10= self.conv10_1(conv9)
        # out = nn.functional.pixel_shuffle(conv10, 2)
        
        if self.adaptive_res_and_x0:
            mask = torch.sigmoid(conv10[:,[0],:,:])
            ## mask  = conv10[:,[0],:,:]  + (conv10[:,[0],:,:].clip(0, 1) - conv10[:,[0],:,:]).detach()
            # mask = 1 -  conv10[:,[0],:,:].clip(0,1)
            
            conv10 = conv10[:,1:]
            out = conv10[:,:self.out_channels//2]
            resid = conv10[:, self.out_channels//2:]
            out_by_resid = (resid+x[:, :self.in_channels]).clip(0,1)
            out_final = out.clip(0,1) * mask + out_by_resid * (1-mask)
            metadata = {"out_by_resid": out_by_resid, "out":out, "mask": mask}
            
        else:
            out = conv10
            if self.resid:
                out = x - out 
            out_final = out.clip(0, 1)        
            metadata = None
        return out_final, metadata
    
        # iter = 1 
        # save_image(out[:,:3], f"d_x0_{iter}.jpg")
        # save_image(resid[:,:3], f"d_res_{iter}.jpg")
        # save_image((resid+x[:, :self.in_channels])[:,:3], f"d_res_x{iter}.jpg")
        # save_image(mask, f"d_mask{iter}.jpg")
        # save_image(conv10[:,:self.out_channels//2][:,:3], f"d_x{iter}.jpg")
        # save_image(x[:, :3], f"d_in{iter}.jpg")
        # save_image(out_final[:, :3], f"d_final{iter}.jpg")
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt
