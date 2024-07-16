import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity
import torch
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.image as mpimg

def run_demo(vis_path, ir_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fusion_model = torch.load(r'model/fusion_model.pth',map_location=torch.device(device))
    deblur_model = torch.load(r'model/deblur_model.pth',map_location=torch.device(device))
    net = torch.load(r'model/att_model.pth',map_location=torch.device(device))
    vis = torch.Tensor(resize(rgb2gray(imread(vis_path)), (640, 640))).unsqueeze(0).unsqueeze(0).to(device)
    ir = torch.Tensor(resize(rgb2gray(imread(ir_path)), (640, 640))).unsqueeze(0).unsqueeze(0).to(device)
    net.to(device)
    fusion_model.to(device)
    deblur_model.to(device)
    fusion_model.eval()
    deblur_model.eval()
    net.eval()

    _, data_deblur = deblur_model(vis, ir)
    data_F = fusion_model(vis, ir, fusion_required=True)
    out = net(data_F, data_deblur, vis, ir)
    mpimg.imsave(r'result_image/{}.jpg'.format(vis_path.split('/')[-1].split('_')[0]), out.detach().cpu().squeeze().numpy() * 255., cmap='gray')
    return out
