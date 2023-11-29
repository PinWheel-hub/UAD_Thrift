import os
import time
import random
import datetime
from random import sample
from collections import OrderedDict
from tqdm import tqdm
import multiprocessing
import shutil
import random
import collections

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

import cv2
from PIL import Image
import yaml
import copy
from skimage import measure, morphology

from uad.datasets import mvtec
from uad.utils.uad_configs import Dic2Obj
from uad.models.patchcore import get_model
from uad.utils.utils_torch import plot_fig

def init_uadetect_func(side_flag=False):
    global model, patchcore_cfg, if_side
    if_side = side_flag
    with open('uad/patchcore.yml', 'r') as file:
        patchcore_cfg = yaml.safe_load(file)
    patchcore_cfg = Dic2Obj(patchcore_cfg)
    random.seed(patchcore_cfg.seed)
    np.random.seed(patchcore_cfg.seed)
    torch.manual_seed(patchcore_cfg.seed)
    torch.cuda.set_device(patchcore_cfg.device[0 if not if_side else 1])
    model = get_model(patchcore_cfg.method)(arch=patchcore_cfg.backbone,
                                    pretrained=False,
                                    k=patchcore_cfg.k,
                                    method=patchcore_cfg.method,
                                    blocks=patchcore_cfg.blocks).cuda()
    state = torch.load(f'xuadetect/models/{patchcore_cfg.backbone}.pth')
    model.model.load_state_dict(state, strict=False)
    model.init_projection()
    model.eval()

##需要观察GPU内存不能泄露，不能一直增长，必要时代码主动释放
def uadetect_func(part, spec_name, part_location, ww, hh):
    global model, patchcore_cfg, if_side
    part = cv2.cvtColor(part, cv2.COLOR_BGR2RGB)
    part = Image.fromarray(part)
    transform = T.Compose([
        T.Resize(patchcore_cfg.resize), T.CenterCrop(patchcore_cfg.crop_size), T.ToTensor(), T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    part = transform(part).unsqueeze(0)
    out = model(part.cuda())
    out = model.project(out)
    scores_map, image_score = model.generate_scores_map(out, part.shape[-2:], spec_name=spec_name)
    # plot_fig(part.numpy(), scores_map, None, patchcore_cfg.threshold, f'{str(part_location[0])}_{str(part_location[1])}_{image_score[0]}.jpg', 'tyre', True)
    threshold = patchcore_cfg.threshold if not if_side else patchcore_cfg.threshold_side
    threshold_sum = patchcore_cfg.threshold_sum  if not if_side else patchcore_cfg.threshold_sum_side
    mask = copy.deepcopy(scores_map[0])
    mask = (mask > threshold) + 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask = np.array(mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = []
    # 遍历每个轮廓
    for i, contour in enumerate(contours):
        # 获取外接矩形框的坐标
        x, y, w, h = cv2.boundingRect(contour)
        score_sum = np.sum(scores_map[0, y: y + h, x: x + w][mask[y: y + h, x: x + w] > 0])
        if score_sum < threshold_sum:
            continue
        score = float(scores_map[0, y: y + h, x: x + w].max())
        if part_location[1] < 1.9 * ww:
            result.append([score, score_sum,
                      [part_location[0] + y * hh / patchcore_cfg.resize[1], part_location[1] + x * ww / patchcore_cfg.resize[0]], 
                      [part_location[0] + (y + h) * hh / patchcore_cfg.resize[1], part_location[1] + (x + w) * ww / patchcore_cfg.resize[0]]])
        else:
            result.append([score, score_sum,
                      [part_location[0] + y * hh / patchcore_cfg.resize[1], part_location[1] + ww - (x + w) * ww / patchcore_cfg.resize[0]], 
                      [part_location[0] + (y + h) * hh / patchcore_cfg.resize[1], part_location[1]  + ww - x * ww / patchcore_cfg.resize[0]]])
    sorted_result = sorted(result, key=lambda item: item[1], reverse=True)
    return sorted_result[:patchcore_cfg.bbox_num]

def load_func(spec_name):
    global model, if_side
    state = torch.load('xuadetect/models/{}.pth'.format(spec_name if not if_side else f'{spec_name}_side'))
    model.load(state["stats"], spec_name)

def del_func():
    global model
    return model.delete_earliest()

def model_exist(spec_name):
    global model
    return model.spec_exist(spec_name)