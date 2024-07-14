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
    # mpimg.imsave(r'result_img/{}'.format(vis_path.split('/')[-1]), out.detach().cpu().squeeze().numpy() * 255., cmap='gray')
    return out


def EN(im):
    histogram, bins = np.histogram(im, bins=256, range=(0, 255))
    histogram = histogram / float(np.sum(histogram))
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy


def SD(im):
    m, n = im.shape
    u = np.mean(im)
    SD = np.sqrt(np.sum(np.sum((im - u) ** 2)) / (m * n))
    return SD


def SF(im):
    RF = np.diff(im, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(im, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF

def MSE(V, I, F):
    V = V / 255.0
    I = I / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_VF = np.sum(np.sum((F - V)**2))/(m*n)
    MSE_IF = np.sum(np.sum((F - I)**2))/(m*n)
    MSE = 0.5 * MSE_VF + 0.5 * MSE_IF
    return MSE


def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r


def SCD(V, I, F):
    r = corr2(F - I, V) + corr2(F - V, I)
    return r

def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2**(4-scale+1)+1
        win = fspecial_gaussian((N, N), N/5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = convolve2d(ref*ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist*dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref*dist, win, mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g*sigma12
        g[sigma1_sq<1e-10] = 0
        sv_sq[sigma1_sq<1e-10] = sigma2_sq[sigma1_sq<1e-10]
        sigma1_sq[sigma1_sq<1e-10] = 0

        g[sigma2_sq<1e-10] = 0
        sv_sq[sigma2_sq<1e-10] = 0

        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=1e-10] = 1e-10

        num += np.sum(np.log10(1+g**2 * sigma1_sq/(sv_sq+sigma_nsq)))
        den += np.sum(np.log10(1+sigma1_sq/sigma_nsq))
    vifp = num/den
    return vifp

def VIF(V, I, F):
    VIF = vifp_mscale(V, F) + vifp_mscale(I, F)
    return VIF


def CC(V, I, F):
    rVF = np.sum((V - np.mean(V)) * (F - np.mean(F))) / np.sqrt(np.sum((V - np.mean(V)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    rIF = np.sum((I - np.mean(I)) * (F - np.mean(F))) / np.sqrt(np.sum((I - np.mean(I)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    CC = np.mean([rVF, rIF])
    return CC

def SSIM(V, I, F):
    return structural_similarity(F, V) + structural_similarity(F, I)