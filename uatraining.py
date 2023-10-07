import os
import sys
import time
import random
import argparse
import datetime
from random import sample
from collections import OrderedDict
from tqdm import tqdm
import multiprocessing
import shutil
import random
import collections

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2
import yaml

from uad.datasets import mvtec
from uad.models.padim import ResNet_PaDiM
from uad.utils.uad_configs import Dic2Obj
from uad.models.patchcore import get_model

# 用padim算法剔除一定比例的离群图片
def padim_del(spec_name, position, cfg):
    # padim初始化
    with open('uad/padim.yml', 'r') as file:
        padim_cfg = yaml.safe_load(file)
    padim_cfg = Dic2Obj(padim_cfg)
    random.seed(padim_cfg.seed)
    np.random.seed(padim_cfg.seed)
    torch.manual_seed(padim_cfg.seed)
    torch.cuda.set_device(padim_cfg.device[0 if position == 'center' else 1])

    model = ResNet_PaDiM(arch=padim_cfg.backbone, pretrained=False).cuda()
    state = torch.load(f'xuadetect/models/{padim_cfg.backbone}.pth')
    model.model.load_state_dict(state)
    model.eval()
    train_dataset = mvtec.MVTecDataset(
        'xuadetect/img_cut/{}/patchcore'.format(spec_name if position == 'center' else f'{spec_name}_side'),
        is_train=True,
        resize=padim_cfg.resize,
        cropsize=padim_cfg.crop_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=padim_cfg.batch_size,
        num_workers=padim_cfg.num_workers)
    fins = {"resnet18": 448, "resnet50": 1792, "wide_resnet50_2": 1792}
    t_d, d = fins[padim_cfg.backbone], 100  # "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4}
    idx = torch.tensor(sample(range(0, t_d), d))

    # padim训练
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # extract train set features
    epoch_begin = time.time()
    end_time = time.time()

    for index, x in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Extracting features"):
        start_time = time.time()
        data_time = start_time - end_time

        # model prediction
        with torch.no_grad():
            outputs = model(x.cuda())

        # get intermediate layer outputs
        for k, v in zip(train_outputs.keys(), outputs):
            train_outputs[k].append(v.cpu().detach())

        end_time = time.time()
        batch_time = end_time - start_time

    del model, outputs
    torch.cuda.empty_cache()
    
    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)
    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        layer_embedding = train_outputs[layer_name]
        layer_embedding = F.interpolate(
            layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
        embedding_vectors = torch.cat((embedding_vectors, layer_embedding), 1)

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx).cuda()
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape((B, C, H * W))
    mean = torch.mean(embedding_vectors, axis=0)
    cov = torch.zeros((C, C, H * W)).cuda()
    I = torch.eye(C).cuda()
    for i in tqdm(range(H * W), desc="Calculating covariance"):
        cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    # save learned distribution
    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(0, t))

    # padim计算距离
    def mahalanobis_pd(sample, mean, conv_inv):
            return torch.sqrt(
                torch.matmul(
                    torch.matmul((sample - mean.unsqueeze(0).expand_as(sample)), conv_inv), (sample - mean.unsqueeze(0).expand_as(sample)).T)).diag()
    dist_list = []
    for i in tqdm(range(H * W), desc="Calculating dsitance"):
        conv_inv = torch.linalg.inv(cov[:, :, i])
        dist = mahalanobis_pd(embedding_vectors[:, :, i], mean[:, i], conv_inv).cpu().numpy()
        dist_list.append(dist)
    
    del mean, cov, embedding_vectors
    torch.cuda.empty_cache()

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(
        dist_list.unsqueeze(1),
        size=x.shape[2:],
        mode='bilinear',
        align_corners=False).squeeze().numpy()
    
    image_score = score_map.reshape(score_map.shape[0], -1).max(axis=1)
    sorted_index = np.argsort(image_score)[::-1]

    del_padim = 'xuadetect/img_cut/{}/del_padim'.format(spec_name if position == 'center' else f'{spec_name}_side')
    if not os.path.exists(del_padim):
        os.makedirs(del_padim)
    del_random = 'xuadetect/img_cut/{}/del_random'.format(spec_name if position == 'center' else f'{spec_name}_side')
    if not os.path.exists(del_random):
        os.makedirs(del_random)

    random_list = list(range(int((1 - cfg["padim_del_rate"]) * len(sorted_index))))
    # 随机打乱下标
    random.shuffle(random_list)
    
    for i, index in enumerate(sorted_index):
        source_path = os.path.join('xuadetect/img_cut/{}/patchcore'.format(spec_name if position == 'center' else f'{spec_name}_side'), train_dataset.img_list[index])
        # print(i - int(cfg["padim_del_rate"] * len(sorted_index)))
        if i < cfg["padim_del_rate"] * len(sorted_index):
            target_path = os.path.join(del_padim, train_dataset.img_list[index])
            shutil.move(source_path, target_path)
        elif random_list[i - int(cfg["padim_del_rate"] * len(sorted_index))] >= cfg["patchcore_train_num"]:
            target_path = os.path.join(del_random, train_dataset.img_list[index])
            shutil.move(source_path, target_path)


    

def train(spec_name, position, cfg):
    # 用padim算法剔除一定比例的离群图片,然后随机保留定量图片
    # padim_del(spec_name, position, cfg)
    print("padim end", torch.cuda.memory_allocated())
    
    with open('uad/patchcore.yml', 'r') as file:
        patchcore_cfg = yaml.safe_load(file)
    
    patchcore_cfg = Dic2Obj(patchcore_cfg)
    random.seed(patchcore_cfg.seed)
    np.random.seed(patchcore_cfg.seed)
    torch.manual_seed(patchcore_cfg.seed)
    torch.cuda.set_device(patchcore_cfg.device[0 if position == 'center' else 1])

    model = get_model(patchcore_cfg.method)(arch=patchcore_cfg.backbone,
                                    pretrained=False,
                                    k=patchcore_cfg.k,
                                    method=patchcore_cfg.method).cuda()
    state = torch.load(f'xuadetect/models/{patchcore_cfg.backbone}.pth')
    model.model.load_state_dict(state, strict=False)
    model.init_projection()
    model.eval()

    # build datasets
    train_dataset = mvtec.MVTecDataset(
        'xuadetect/img_cut/{}/patchcore'.format(spec_name if position == 'center' else f'{spec_name}_side'),
        is_train=True,
        resize=patchcore_cfg.resize,
        cropsize=patchcore_cfg.crop_size,
        max_size=patchcore_cfg.max_size if hasattr(patchcore_cfg, 'max_size') else None)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=patchcore_cfg.batch_size,
        num_workers=patchcore_cfg.num_workers)

    epoch_begin = time.time()
    # extract train set features
    outs = []
    for x in tqdm(train_dataloader, '| feature extraction | train |'):
        # model prediction
        out = model(x.cuda())
        out = model.project(out)
        outs.append(out)
    del out, x
    torch.cuda.empty_cache()
    outs = torch.concat(outs, 0)
    C = outs.shape[1]
    outs = outs.permute((0, 2, 3, 1)).reshape((-1, C))
    model.compute_stats(outs)
    del outs
    torch.cuda.empty_cache()

    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(t))
    print("Saving model...")
    save_name = 'xuadetect/models/{}.pth'.format(spec_name if position == 'center' else f'{spec_name}_side')
    dir_name = os.path.dirname(save_name)
    os.makedirs(dir_name, exist_ok=True)
    memory_bank = collections.OrderedDict()
    memory_bank['memory_bank'] =  model.memory_bank
    state_dict = {"stats": memory_bank,}
    torch.save(state_dict, save_name)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Save model in {}".format(str(save_name)))

def TrainingProc(spec_name, cfg):
    # 分割处理图片
    # img_dir = f'xuadetect/img_raw/{spec_name}'
    # img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    # save_dir = f'xuadetect/img_cut/{spec_name}/patchcore'
    # save_side_dir = f'xuadetect/img_cut/{spec_name}_side/patchcore'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # if not os.path.exists(save_side_dir):
    #     os.makedirs(save_side_dir)
    # col_num = 3
    # for img_num, img_file in enumerate(img_files):
    #     img = cv2.imread(os.path.join(img_dir, img_file))
    #     i = 0
    #     patch_length = img.shape[1] // col_num
    #     row_num = img.shape[0] // patch_length
    #     for i in range(0, row_num + 1):
    #         for j in range(0, col_num):
    #             if patch_length * (i + 1) < img.shape[0]:
    #                 rg = img[patch_length * i: patch_length * (i + 1), patch_length * j: patch_length * (j + 1)]
    #             elif img.shape[0] - patch_length * i > patch_length / 2:
    #                 rg = img[img.shape[0] - patch_length: img.shape[0], patch_length * j: patch_length * (j + 1)]
    #             else:
    #                 break
    #             cv2.imwrite(os.path.join(save_dir if j == 1 else save_side_dir, f'{os.path.splitext(img_file)[0]}_{j}_{i}.jpg'), rg if j < 2 else cv2.flip(rg, 1))

    process = multiprocessing.Process(target=train, args=(spec_name, 'center', cfg))
    process_side = multiprocessing.Process(target=train, args=(spec_name, 'side', cfg))
    process.start()
    process_side.start()
    process.join()
    process_side.join()

    if not os.path.exists('xuadetect/loadlist') and os.path.exists(f'xuadetect/models/{spec_name}.pth') and os.path.exists(f'xuadetect/models/{spec_name}_side.pth'):
        os.makedirs('xuadetect/loadlist/')
    with open(f'xuadetect/loadlist/{spec_name}', "w") as f:
        pass

    if os.path.exists(f'xuadetect/trainlist/{spec_name}'):
        os.remove(f'xuadetect/trainlist/{spec_name}')

    