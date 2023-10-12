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

from uad.datasets import mvtec
from uad.utils.uad_configs import Dic2Obj
from uad.models.patchcore import get_model

with open('uad/patchcore.yml', 'r') as file:
    patchcore_cfg = yaml.safe_load(file)
patchcore_cfg = Dic2Obj(patchcore_cfg)
model = None

def init_uadetect_func(if_side):
    global model
    random.seed(patchcore_cfg.seed)
    np.random.seed(patchcore_cfg.seed)
    torch.manual_seed(patchcore_cfg.seed)
    torch.cuda.set_device(patchcore_cfg.device[0 if not if_side else 1])
    model = get_model(patchcore_cfg.method)(arch=patchcore_cfg.backbone,
                                    pretrained=False,
                                    k=patchcore_cfg.k,
                                    method=patchcore_cfg.method).cuda()
    state = torch.load(f'xuadetect/models/{patchcore_cfg.backbone}.pth')
    model.model.load_state_dict(state, strict=False)
    model.init_projection()
    model.eval()

##需要观察GPU内存不能泄露，不能一直增长，必要时代码主动释放
def uadetect_func(part, spec_name, part_location):
    global model
    part = cv2.cvtColor(part, cv2.COLOR_BGR2RGB)
    part = Image.fromarray(part)
    transform = T.Compose([
        T.Resize(patchcore_cfg.resize), T.CenterCrop(patchcore_cfg.crop_size), T.ToTensor(), T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    part = transform(part).unsqueeze(0)
    out = model(part.cuda())
    out = model.project(out)
    score_map, image_score = model.generate_scores_map(out, part.shape[-2:], spec_name=spec_name)
    return image_score[0], part_location, image_score[0] > patchcore_cfg.threshold

def load_func(spec_name, if_side):
    global model
    state = torch.load('xuadetect/models/{}.pth'.format(spec_name if not if_side else f'{spec_name}_side'))
    model.load(state["stats"], spec_name)

def del_func():
    global model
    return model.delete_earliest()