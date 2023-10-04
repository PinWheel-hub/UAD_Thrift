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

def train(spec_name, position, cfg):
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

    train_dataset = mvtec.MVTecDataset(
        'xuadetect/img_cut/{}/del_padim'.format(spec_name if position == 'center' else f'{spec_name}_side'),
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

    for index, x in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
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
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape((B, C, H * W))
    mean = torch.mean(embedding_vectors, axis=0).numpy()
    cov = torch.zeros((C, C, H * W)).numpy()
    I = np.identity(C)
    for i in tqdm(range(H * W)):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    # save learned distribution
    train_outputs = [torch.tensor(mean), torch.tensor(cov)]
    model.distribution = train_outputs
    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(0, t))
    
    # save_name = 'xuadetect/models/padim_{}.pth'.format(spec_name if position == 'center' else f'{spec_name}_side')
    # dir_name = os.path.dirname(save_name)
    # if dir_name and not os.path.exists(dir_name):
    #     os.makedirs(dir_name)
    # state_dict = {
    #     "params": model.model.state_dict(),
    #     "distribution": model.distribution,}
    # torch.save(state_dict, save_name)
    # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Save model in {}".format(str(save_name)))

    # 用padim算法剔除一定比例的离群图片
    model.eval()
    def mahalanobis_pd(sample, mean, conv_inv):
            return torch.sqrt(torch.matmul(torch.matmul((sample - mean).unsqueeze(1).T, conv_inv), (sample - mean)))[0]

    embedding_vectors = embedding_vectors.reshape((B, C, H * W)).cuda()
    model.distribution[0] = model.distribution[0].cuda()
    model.distribution[1] = model.distribution[1].cuda()
    dist_list = []
    for i in range(H * W):
        mean = model.distribution[0][:, i]
        conv_inv = torch.linalg.inv(model.distribution[1][:, :, i])
        dist = [mahalanobis_pd(sample[:, i], mean, conv_inv).cpu().numpy()
                for sample in embedding_vectors]
        dist_list.append(dist)


    with open('uad/patchcore.yml', 'r') as file:
        patchcore_cfg = yaml.safe_load(file)

def TrainingProc(spec_name, cfg):
    # # 分割处理图片
    # img_dir = f'xuadetect/img_raw/{spec_name}'
    # img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    # save_dir = f'xuadetect/img_cut/{spec_name}/del_padim'
    # save_side_dir = f'xuadetect/img_cut/{spec_name}_side/del_padim'
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

    # if os.path.exists(f'xuadetect/trainlist/{spec_name}'):
    #     os.remove(f'xuadetect/trainlist/{spec_name}')

    