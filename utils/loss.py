import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import torchvision.models as models


class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.sobelconv=Sobelxy()
        self.L1Loss = nn.L1Loss()

    def forward(self,V,I,G):
        image_y=V[:,:1,:,:]
        x_in_max=torch.max(image_y,I)
        loss_in=F.l1_loss(x_in_max,G)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(I)
        G_grad=self.sobelconv(G)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,G_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.sobelconv=Sobelxy()
        self.loss_SSIMLoss = kornia.losses.SSIMLoss(11, reduction='mean')
        self.L1Loss = nn.L1Loss()

    def forward(self,V,I,G):
        image_y=V[:,:1,:,:]
        x_in_max=torch.max(image_y,I)
        loss_in=F.l1_loss(x_in_max,G)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(I)
        G_grad=self.sobelconv(G)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,G_grad)
        # loss_total=loss_in+10*loss_grad+self.loss_SSIMLoss(V, G)
        # gradient_loss = self.L1Loss(kornia.filters.SpatialGradient()(V),
        #                            kornia.filters.SpatialGradient()(G))
        ssim_loss = self.loss_SSIMLoss(V, G)
        loss_total=10*loss_grad+ssim_loss+5*loss_in
        # print(10*gradient_loss, ssim_loss, loss_in)
        # loss_total=loss_in+5*self.loss_SSIMLoss(V, G)
        return loss_total
    
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.SSIMLoss_loss = kornia.losses.SSIMLoss(11, reduction='mean')
        self.mse_loss = nn.MSELoss()

    def forward(self, hat, target):
        return 5 * self.SSIMLoss_loss(target, hat) + self.mse_loss(target, hat)


class DeblurLoss(nn.Module):
    def __init__(self):
        super(DeblurLoss, self).__init__()
        self.SSIMLoss_loss = kornia.losses.SSIMLoss(11, reduction='mean')
        self.perceptual_loss = PerceptualLoss()

    def forward(self, db, vis, ir, i):
        # SSIMLoss_loss, mse_loss_ir, mse_loss_vis = self.SSIMLoss_loss(vis, db), F.mse_loss(ir, db), F.mse_loss(vis, db)
        SSIMLoss_loss = self.SSIMLoss_loss(vis, db)
        # p1 = self.perceptual_loss(db, ir, 0)
        p2 = self.perceptual_loss(db, ir, i)
        # p3 = self.perceptual_loss(db, ir, 2)
        loss = SSIMLoss_loss + 100*p2
        return loss


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vgg = self.get_vgg().to(device)
        # self.vgg = self.get_vgg()
        self.bns = [i - 2 for i, m in enumerate(self.vgg) if isinstance(m, nn.MaxPool2d)]
        # self.feature_extractor = [self.model_hook(i) for i in self.bns]
    
    def get_vgg(self):
        vgg = models.vgg16_bn(pretrained=True)
        vgg.features[0]=nn.Conv2d(1, 64, kernel_size=3, padding=1)
        vgg = vgg.features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        return vgg

    def model_hook(self, hook_layer):
        model = nn.Sequential()
        model = model.cuda()

        for i,layer in enumerate(list(self.vgg)):
            model.add_module(str(i),layer)
            if i == hook_layer:
                break
        return model
    
    def forward(self, input, target_i, i):
        model = self.model_hook(self.bns[i])
        fin = model.forward(input)
        fi = model.forward(target_i)
        return fin, fi
        # model = self.model_hook(self.bns[i])
        # fin = model.forward(input)
        # fv = self.feature_extractor[i].forward(target_v)
        # fi = model.forward(target_i)
        # loss = F.mse_loss(fi, fin)
        # return loss