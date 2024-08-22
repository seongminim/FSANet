from __future__ import print_function
import argparse
import torch
from model import FSA
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
import math
from lib.dataset import is_image_file
from PIL import Image
from os import listdir
from skimage import color, filters
from ptflops import get_model_complexity_info
from thop import profile
import os
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=16, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default='test_img')
parser.add_argument('--model_type', type=str, default='FSA')
parser.add_argument('--output', default='./output/', help='Location to save checkpoint models')
parser.add_argument('--modelfile', default='models/FSA_test.pth', help='sr pretrained base model')
parser.add_argument('--image_based', type=bool, default=True, help='use image or video based ULN')
parser.add_argument('--chop', type=bool, default=False)

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Building model ', opt.model_type)

model = FSA()
model = torch.nn.DataParallel(model, device_ids=gpus_list)
model.load_state_dict(torch.load(
    opt.modelfile,
    map_location=lambda storage, loc: storage))
if cuda:
    model = model.cuda(gpus_list[0])
    model = model.type('torch.cuda.HalfTensor')


def eval():
    model.eval()
    all_datas = ['LOL', 'MEF', 'DICM', 'LIME', 'VV']
    for all_data in all_datas:
        LL_filename = os.path.join(opt.image_dataset, all_data)
        est_filename = os.path.join(opt.output, all_data)
        test_NL_folder = "datasets/LOL/test/high/"
        try:
            os.stat(est_filename)
        except:
            os.mkdir(est_filename)
        LL_image = [join(LL_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
        test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_NL_folder)) if is_image_file(x)]
        print(LL_image)
        Est_img = [join(est_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
        print(Est_img)
        trans = transforms.ToTensor()
        channel_swap = (1, 2, 0)
        time_ave = 0

        for i in range(LL_image.__len__()):
            with torch.no_grad():
                LL_in = Image.open(LL_image[i]).convert('RGB')
                LL_int = trans(LL_in)
                c, h, w = LL_int.shape
                h_tmp = (h % 4)
                w_tmp = (w % 4)
                LL_in = LL_in.crop((0, 0, w - w_tmp, h - h_tmp))
                LL = trans(LL_in)
                LL = LL.unsqueeze(0)
                LL = LL.cuda()
                LL = LL.type('torch.cuda.HalfTensor')
                t0 = time.time()
                prediction = model(LL)
                t1 = time.time()
                time_ave += (t1 - t0)
                prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
                prediction = prediction * 255.0
                prediction = prediction.clip(0, 255)
                Image.fromarray(np.uint8(prediction)).save(Est_img[i])
                print("===> Processing Image: %04d /%04d in %.4f s." % (i, LL_image.__len__(), (t1 - t0)))

        print("===> Processing Time: %.4f ms." % (time_ave / LL_image.__len__() * 1000))


transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
]
)

# Eval Start!!!!
eval()
