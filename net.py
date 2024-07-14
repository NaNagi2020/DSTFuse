import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Net, self).__init__()
        self.att = Attention(feature_dim=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, fusion, deblur, vis, ir):
        att = self.att(ir, ir.shape[-1])
        return self.sigmoid((deblur+vis)*att + (fusion+vis)*(1-att))


class FusionNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(FusionNet, self).__init__()
        # self.inblock1 = EBlock(1+1, 32, 1)
        self.inblock2 = EBlock(1, 32, 1)
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        self.fusion_layer = FusionLayer()
        self.dblock1 = DBlock(128, 64, 2)
        self.dblock2 = DBlock(64, 32, 2)
        self.outblock = OutBlock(32)
        self.sigmoid = nn.Sigmoid()
        self.sobel = SobelConvLayer()

    def forward_fusion(self, vis, ir):
        e32_v = self.inblock2(vis)
        e64_v = self.eblock1(e32_v)
        e128_v = self.eblock2(e64_v)
        e32_i = self.inblock2(ir)
        e64_i = self.eblock1(e32_i)
        e128_i = self.eblock2(e64_i)

        d64 = self.dblock1(self.fusion_layer(e128_v+e128_i, ir))
        d32 = self.dblock2(d64 + e64_v)
        d1 = self.outblock(d32 + e32_v)
        out = self.sigmoid(d1+vis)
        return out
    
    def forward_reconstruction(self, vis, ir):
        x = vis if vis is not None else ir
        e32 = self.inblock2(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        d64 = self.dblock1(e128)
        d32 = self.dblock2(d64 + e64)
        d1 = self.outblock(d32 + e32)
        return self.sigmoid(d1+x)
    
    def forward(self, vis, ir, fusion_required=False):
        data_v, data_i = vis, ir
        if fusion_required:
            return self.forward_fusion(data_v, data_i)
        return self.forward_reconstruction(vis=data_v, ir=data_i), self.forward_reconstruction(vis=None, ir=data_i)


class DeblurFuse(nn.Module):
    def __init__(self):
        super(DeblurFuse, self).__init__()
        self.inblock1 = EBlock(1+1, 32, 1)
        self.inblock2 = EBlock(1, 32, 1)
        self.conv1 = ConvLayer(1, 64, 3, 2) #16
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        self.dblock1 = DBlock(128, 64, 2)
        self.dblock2 = DBlock(64, 32, 2)
        self.outblock = OutBlock(32)
        self.sigmoid = nn.Sigmoid()
        self.sobel = SobelConvLayer()

    def forward_step(self, b, vis, ir, is_last=False): 
        data_sobel = self.sobel(ir,torch.ones_like(b))
        e32 = self.inblock2(b) if is_last else self.inblock1(torch.cat([b, self.sigmoid(vis+data_sobel)], dim=1))
        # feature_I = None if is_last else self.dense_block(self.conv1(ir))
        # feature_S = None if is_last else self.extractor_i(self.scale(vis+data_sobel))
        feature_S = self.conv1(vis+data_sobel)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        d64 = self.dblock1(e128)
        d32 = self.dblock2(d64 + e64 + feature_S)
        d = self.outblock(d32 + e32)
        # out = self.sigmoid(d3+vis) if is_last else self.sigmoid(d3+ir)
        # return out, d3
        return d
    
    def forward(self, b, ir):
        vis = b
        # d1, o1 = self.forward_step(b, vis, ir)
        # d3, o3 = self.forward_step(d1, vis, ir, is_last=True)
        d1 = self.forward_step(b, vis, ir)
        d2 = self.forward_step(d1, vis, ir, is_last=True)
        out = self.sigmoid(d2+d1+b)
        # return d1, d3, o1, o3
        return d1, out


class FusionLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(FusionLayer, self).__init__()
        self.attention = Attention(feature_dim=1)
        self.norm = nn.BatchNorm2d(in_channels)
    
    def forward(self, x, ir):
        # out = x + self.attention(self.norm(x), vis, ir, mask)
        out = x + x * (1-self.attention(ir, size=x.shape[-1]))
        # out = x + self.norm(self.attention(x, vis, ir, mask))
        return out


class SobelConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SobelConvLayer, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
    
    def forward(self, ir, mask):
        sobelx=F.conv2d(ir*mask, self.weightx, padding=1)
        sobely=F.conv2d(ir*mask, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely) # [1,1,640,640]



class Attention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_dim=64, conv=nn.Conv2d, norm=nn.BatchNorm2d, act=nn.ELU):
        super(Attention, self).__init__()
        self.att_bolck = nn.Sequential(
            conv(feature_dim,in_channels, 3, stride=1, padding=1),
            norm(in_channels, eps=0.00001, momentum=0.1),
            act(),
            conv(in_channels,in_channels,3, stride=1, padding=1),
            norm(in_channels, eps=0.00001, momentum=0.1),
            act()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=1)

    def forward(self, x, size):
        x1 = self.att_bolck(x)
        x2 = torch.sigmoid(self.conv1(x1))
        att_map = F.interpolate(x2, size=size, mode='bilinear')
        out = att_map
        return out
    
class SimpleExtractor(nn.Module):
    def __init__(self):
        super(SimpleExtractor, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.block(x)
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
#             out = F.normalize(out)
            out = F.relu(out, inplace=True)
#             out = self.dropout(out)
        return out


class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, 5, 1)
        self.conv2 = ConvLayer(out_channels, out_channels, 5, 1, is_last=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = x2 + x
        return out


class EBlock(nn.Module):
    def __init__( self , in_channels , out_channels, stride):
        super(EBlock, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels, 5, stride)

        resblock_list = []
        for i in range( 3):
            resblock_list.append(ResBlock(out_channels, out_channels))
        self.resblock_stack = nn.Sequential( *resblock_list )

    def forward( self , x):
        x = self.conv(x)
        x = self.resblock_stack(x)
        return x

class DBlock(nn.Module):
    def __init__( self , in_channels , out_channels, stride):
        super(DBlock, self).__init__()
        resblock_list = []
        for i in range( 3):
            resblock_list.append(ResBlock(in_channels, in_channels))
        self.resblock_stack = nn.Sequential( *resblock_list )
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 5, stride, 2, 1)

    def forward( self , x ):
        x = self.resblock_stack( x )
        x = self.deconv( x )
        return x

class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(OutBlock, self).__init__()
        resblock_list = []
        for i in range( 3):
            resblock_list.append(ResBlock(in_channels, in_channels))
        self.resblock_stack = nn.Sequential( *resblock_list )
        self.conv = ConvLayer(in_channels, 1, 5, 1, is_last=True) 
    def forward( self , x ):
        x = self.resblock_stack( x )
        x = self.conv( x )
        return x