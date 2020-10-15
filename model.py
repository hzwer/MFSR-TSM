import megengine as mge
import megengine.module as M
import megengine.functional as F
from megengine.core import Parameter
from utils import *

def addLeakyRelu(x):
    return M.Sequential(x, M.LeakyReLU(0.1))

def addSig(x):
    return M.Sequential(x, M.Sigmoid())

def up_block(x, ic, oc):
    return M.ConvTranspose2d(ic, oc, 4, stride=2, padding=1)

def down_block(x, ic, oc):
    return M.Conv2d(ic, oc, 3, padding=1, stride=2)

class BasicBlock(M.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm=M.BatchNorm2d,
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = M.Conv2d(
            in_channels, channels, 3, stride, padding=dilation, bias=True
        )
        self.conv2 = M.Conv2d(channels, channels, 3, 1, padding=1, bias=True)
        if in_channels == channels and stride == 1:
            self.downsample = M.Identity()
        elif stride == 1:
            self.downsample = M.Conv2d(in_channels, channels, 1, stride, bias=False)
        else:
            self.downsample = M.Sequential(
                    M.AvgPool2d(kernel_size=stride, stride=stride),
                    M.Conv2d(in_channels, channels, 1, 1, bias=False)
                )
        self.fc1 = M.Conv2d(channels, 16, kernel_size=1) 
        self.fc2 = M.Conv2d(16, channels, kernel_size=1)
        self.relu1 = M.LeakyReLU(0.1)
        self.relu2 = M.LeakyReLU(0.1)
        self.relu3 = M.LeakyReLU(0.1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        identity = self.downsample(identity)
        w = x.mean(3, True).mean(2, True)
        w = self.relu2(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        x = x * w + identity
        x = self.relu3(x)
        return x

def subpixel(x):
    shape = x.shape
    x = x.reshape(shape[0], shape[1] // 4, 2, 2, shape[2], shape[3])
    x = F.dimshuffle(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(shape[0], shape[1] // 4, shape[2]*2, shape[3]*2)

c = 64
class SimpleUNet(M.Module):
    def __init__(self):
        super().__init__()

        self.conv0_ = (BasicBlock(3, 32, stride=2))
        self.conv1_ = (BasicBlock(32, c, stride=2))
        self.conv0 = (BasicBlock(15, 32, stride=2))
        self.conv1 = (BasicBlock(32, c, stride=2))
        self.conv2 = (BasicBlock(c, 2*c, stride=1))
        self.conv3 = (BasicBlock(2*c, 2*c, stride=1))
        self.conv4 = (BasicBlock(4*c, 2*c, stride=1))
        self.conv5 = (BasicBlock(4*c, 2*c, stride=1))
        self.conv6 = (BasicBlock(6*c, 2*c, stride=1))
        self.conv7 = (BasicBlock(6*c, 2*c, stride=1))
        self.conv8 = (BasicBlock(6*c, 2*c, stride=1))
        self.conv9 = (BasicBlock(6*c, 2*c, stride=1))
        self.conv10 = (BasicBlock(3*c, 4*c, stride=1))
        self.conv11 = addSig(M.Conv2d(c+32, 12, 1))

    def forward(self, x):
        x = x[:, 6:21]
        size = x.shape
        x = x.reshape((size[0] * 5, 3) + size[2:])
        conv0 = tsm(self.conv0_(x))
        conv1 = tsm(self.conv1_(conv0))
        #
        x = (x.reshape((size[0], 15) + x.shape[2:]))
        conv0_ = (conv0.reshape((size[0], 5) + conv0.shape[1:]))[:, 2]
        conv1_ = (conv1.reshape((size[0], 5) + conv1.shape[1:]))[:, 2]
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv0 += conv0_
        conv1 += conv1_
        
        conv2 = (self.conv2(conv1))
        conv3 = (self.conv3(conv2))
        conv4 = (self.conv4(F.concat((conv3, conv2), 1)))
        conv5 = (self.conv5(F.concat((conv4, conv3), 1)))
        conv6 = (self.conv6(F.concat((conv5, conv4, conv2), 1)))
        conv7 = (self.conv7(F.concat((conv6, conv5, conv3), 1)))
        conv8 = (self.conv8(F.concat((conv7, conv6, conv4), 1)))
        conv9 = (self.conv9(F.concat((conv8, conv7, conv5), 1)))
        conv10 = subpixel(self.conv10(F.concat((conv9, conv1), 1)))
        conv11 = subpixel(self.conv11(F.concat((conv10, conv0), 1))) 
        conv11 = conv11 * 2 - 1 # sigmoid to [-1, 1]

        return F.minimum(F.maximum(conv11 + x[:, 6:9], 0), 1)
