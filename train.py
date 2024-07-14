import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kornia

from net import FusionNet, DeblurFuse, Net
from dataset import H5Dataset
from utils.loss import DeblurLoss, FusionLoss, ReconstructionLoss, AttentionLoss
from utils.log import get_logger
from detector import Detector


lr = 1e-4
weight_decay = 0
batch_size = 2
optim_step = 20
optim_gamma = 0.5
num_epochs = 15
epoch_gap = 10
# num_epochs = 6
# epoch_gap = 0
fusion_gap = 6
reconstruct_gap = 3
im_size = 640

logger = get_logger(r'model/exp.log')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(device)
dataset = H5Dataset(os.path.join('data', "LLVIP_blur_train"+'.h5'))
trainloader = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0,
                         drop_last=True)

fusion_model = FusionNet()
deblur_model = DeblurFuse()
# fusion_model = torch.load(r'model/fusion_model.pth')
# deblur_model = torch.load(r'model/deblur_model.pth')
att_model = Net()

optimizer1 = torch.optim.Adam([p for p in fusion_model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam([p for p in deblur_model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam([p for p in att_model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

criteria_fusion = FusionLoss()
criteria_deblur = DeblurLoss()
criteria_reconstruction = ReconstructionLoss()
criteria_att = AttentionLoss()

# det_model.to(device)
fusion_model.to(device)
deblur_model.to(device)
att_model.to(device)
labels = dataset.get_label()

for epoch in range(num_epochs):
    for i, (index, vis, ir) in enumerate(trainloader):
        fusion_model.zero_grad()
        deblur_model.zero_grad()
        att_model.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
            
        vis, ir = vis.to(device), ir.to(device) # tensor [1, 1, 640, 640]
        
        if epoch < epoch_gap: # phase 1
            deblur_model.train()
            fusion_model.train()
            if epoch < reconstruct_gap:
                vis_hat, ir_hat = fusion_model(vis, ir, fusion_required=False)
                loss_V = criteria_reconstruction(vis_hat, vis)
                loss_I = criteria_reconstruction(ir_hat, ir)
    #             gradient_loss = L1Loss(kornia.filters.SpatialGradient()(vis),
    #                                    kornia.filters.SpatialGradient()(vis_hat))
                reconstruct_loss = loss_V + loss_I
                reconstruct_loss.backward()
                optimizer1.step()
                if i % 10 == 0:
                    logger.info('Epoch:{}\titer:{}\treconstruct_loss:{:.5f}'.format(
                        epoch, i, reconstruct_loss))
            elif epoch < fusion_gap:
                data_fusion = fusion_model(vis, ir, fusion_required=True)
                fusionloss = criteria_fusion(vis, ir, data_fusion)
                fusionloss.backward()
                optimizer1.step()
                if i % 10 == 0:
                    logger.info("epoch:{}\titer:{}\tfusion_loss:{:.5f}".format(epoch, i, fusionloss))

            d1, d2 = deblur_model(vis, ir)
            d1_loss = criteria_deblur(d1, vis, ir, 3)
            d2_loss = criteria_deblur(d2, vis, ir, 2)
            deblur_loss = d1_loss+d2_loss
            deblur_loss.backward()
            optimizer2.step()
            if i % 10 == 0:
                logger.info("epoch:{}\titer:{}\tdeblur_loss:{:.5f}".format(epoch, i, deblur_loss))
            
            
        else : #phase 2
            deblur_model.eval()
            fusion_model.eval()
            with torch.no_grad():
                _, data_deblur = deblur_model(vis, ir)
                data_fusion = fusion_model(vis, ir, fusion_required=True)
            out = att_model(data_fusion, data_deblur, vis, ir)
            attloss = criteria_att(vis, ir, out)
            attloss.backward()
            optimizer3.step()
            if i % 10 == 0:
                logger.info("epoch:{}\titer:{}\tnet_loss:{:.5f}".format(epoch, i, attloss))

    torch.save(fusion_model, r"model/fusion_model.pth")
    torch.save(deblur_model, r"model/deblur_model.pth")
    torch.save(att_model, r"model/att_model.pth")
