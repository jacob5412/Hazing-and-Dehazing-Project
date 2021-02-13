import torch
import torch.nn as nn
from torchvision.models import vgg16


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, batch):
        return batch.view([batch.shape[0], -1])


class Scale1_Linear(nn.Module):
    #input 512x7x10
    #output 64x15x20
    
    def __init__(self):
        super(Scale1_Linear, self).__init__()
        self.block = nn.Sequential(
            Flatten(),
            nn.Linear(512*7*10, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 64*15*20)
        )
    
    def forward(self, x):
        scale_1_op = torch.reshape(self.block(x), (x.shape[0], 64, 15, 20))
        return nn.functional.interpolate(scale_1_op, scale_factor=4, mode='bilinear', align_corners=True)


class Scale2(nn.Module):
    #input 64x60x80, 3x240x320
    #output 1x120x160
    
    def __init__(self):
        super(Scale2, self).__init__()
        self.input_img_proc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        )
    
    def forward(self, x, input_img):
        proc_img = self.input_img_proc(input_img)
        concatenate_input = torch.cat((x,proc_img), dim=1)
        return nn.functional.interpolate(self.block(concatenate_input), scale_factor=2, mode='bilinear', align_corners=True)


class Scale3(nn.Module):
    #input 1x120x160, 3x240x320
    #output 1x120x160
    
    def __init__(self):
        super(Scale3, self).__init__()
        self.input_img_proc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        )
    
    def forward(self, x, input_img):
        proc_img = self.input_img_proc(input_img)
        concatenate_input = torch.cat((x,proc_img), dim=1)
        return self.block(concatenate_input)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.VGG = nn.Sequential(*list(vgg16(pretrained=True).children())[0])
        self.Scale_1 = Scale1_Linear()
        self.Scale_2 = Scale2()
        self.Scale_3 = Scale3()
    
    def forward(self, x):
        input_img = x.clone()                  # 3x240x320
        x = self.VGG(x)                        # 512x7x10
        x = self.Scale_1(x)                    # 64x60x80
        x = self.Scale_2(x, input_img.clone()) # 1x120x160
        x = self.Scale_3(x, input_img.clone()) # 1x120x160
        return x
