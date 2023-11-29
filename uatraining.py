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
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml

from uad.datasets import mvtec
from uad.models.padim import ResNet_PaDiM
from uad.utils.uad_configs import Dic2Obj
from uad.models.patchcore import get_model


# 用padim算法剔除一定比例的离群图片
def padim_sort(spec_name, if_side, cfg):
    # padim初始化
    with open('uad/padim.yml', 'r') as file:
        padim_cfg = yaml.safe_load(file)
    padim_cfg = Dic2Obj(padim_cfg)
    random.seed(padim_cfg.seed)
    np.random.seed(padim_cfg.seed)
    torch.manual_seed(padim_cfg.seed)
    torch.cuda.set_device(padim_cfg.device[0 if not if_side else 1])

    model = ResNet_PaDiM(arch=padim_cfg.backbone, pretrained=False).cuda()
    state = torch.load(f'xuadetect/models/{padim_cfg.backbone}.pth')
    model.model.load_state_dict(state)
    model.eval()
    train_dataset = mvtec.MVTecDataset(
        'xuadetect/img_cut/{}/padim'.format(spec_name if not if_side else f'{spec_name}_side'),
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

    del outputs
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
    del train_outputs, layer_embedding

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx).cuda()
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape((B, C, H * W))
    mean = torch.mean(embedding_vectors, axis=0)
    cov = torch.zeros((C, C, H * W)).cuda()
    I = torch.eye(C).cuda()
    for i in range(H * W):
        cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    # save learned distribution
    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(0, t))
    del embedding_vectors
    torch.cuda.empty_cache()
    raw_sel = 'xuadetect/img_cut/{}/raw_sel'.format(spec_name if not if_side else f'{spec_name}_side')
    img_files = os.listdir(raw_sel)
    results = {}
    img_dic = {}
    transform_x = train_dataset.get_transform_x()
    for img_file in tqdm(img_files, desc="Calculating scores"):
        x = Image.open(os.path.join(raw_sel, img_file)).convert('RGB')
        x = transform_x(x).unsqueeze(0)
        
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # model prediction
        with torch.no_grad():
            outputs = model(x.cuda())
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v)
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            layer_embedding = test_outputs[layer_name]
            layer_embedding = F.interpolate(
                layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
            embedding_vectors = torch.cat((embedding_vectors, layer_embedding), 1)

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx.cuda())

        # calculate distance matrix
        B, C, H, W = embedding_vectors.shape
        embedding = embedding_vectors.reshape((B, C, H * W))
        inv_covariance = torch.linalg.inv(cov.permute(2, 0, 1))

        delta = (embedding - mean).permute((2, 0, 1))

        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute((1, 0))
        distances = distances.reshape((B, H, W))
        distances = torch.sqrt(distances)

        max_score = distances.max()
        index = img_file.rfind('_', 0, img_file.rfind('_'))
        img = img_file[:index]
        if img not in results:
            results[img] = []
        if img not in img_dic:
            img_dic[img] = []
        results[img].append(max_score)
        img_dic[img].append(img_file)
    for key in results:
        results[key] = sum(results[key]) / len(results[key])
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
    del mean, cov
    torch.cuda.empty_cache()
    return sorted_results, img_dic

def trainset_generate(spec_name, if_side, cfg, sorted_results={}, img_dic={}):
    patchcore_trainset = 'xuadetect/img_cut/{}/patchcore'.format(spec_name if not if_side else f'{spec_name}_side')
    if not os.path.exists(patchcore_trainset):
        os.makedirs(patchcore_trainset)
    raw_sel = 'xuadetect/img_cut/{}/raw_sel'.format(spec_name if not if_side else f'{spec_name}_side')
    i = 0
    if len(sorted_results) > 0 and len(img_dic) > 0:
        for key in sorted_results:
            for img_file in img_dic[key]:
                if i >= cfg['patchcore_train_num']:
                    return
                source_path = os.path.join(raw_sel, img_file)
                target_path = os.path.join(patchcore_trainset, img_file)
                shutil.copy(source_path, target_path)
                i += 1
    else:
        raw_imgs = os.listdir(raw_sel)
        raw_imgs = [f for f in raw_imgs if f.endswith('.jpg') or f.endswith('.png')]
        for img_file in raw_imgs:
            if i >= cfg['patchcore_train_num']:
                    return
            source_path = os.path.join(raw_sel, img_file)
            target_path = os.path.join(patchcore_trainset, img_file)
            shutil.copy(source_path, target_path)
            i += 1


# def random_del(spec_name, if_side, cfg):
#     del_random = 'xuadetect/img_cut/{}/del_random'.format(spec_name if not if_side else f'{spec_name}_side')
#     if not os.path.exists(del_random):
#         os.makedirs(del_random)
#     source_dir = 'xuadetect/img_cut/{}/patchcore'.format(spec_name if not if_side else f'{spec_name}_side')
#     files = os.listdir(source_dir)
#     imgs = [file for file in files if file.endswith(".jpg") or file.endswith(".png")]
#     random_list = list(range(len(imgs)))
#     # 随机打乱下标
#     random.shuffle(random_list)
#     for i, img in enumerate(imgs):
#         if random_list[i] >= cfg["patchcore_train_num"]:
#             source_path = os.path.join(source_dir, img)
#             target_path = os.path.join(del_random, img)
#             shutil.move(source_path, target_path)    

def train(spec_name, if_side, cfg):
    # 用padim算法剔除一定比例的离群图片,然后随机保留定量图片
    sorted_results, img_dic = padim_sort(spec_name, if_side, cfg)
    trainset_generate(spec_name, if_side, cfg, sorted_results, img_dic)
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

    # build datasets
    train_dataset = mvtec.MVTecDataset(
        'xuadetect/img_cut/{}/patchcore'.format(spec_name if not if_side else f'{spec_name}_side'),
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
    B, C, H, W = outs.shape
    if patchcore_cfg.method == 'local_coreset':  
        outs = outs.permute((2, 3, 0, 1)).reshape((model.blocks[0], 
                                                            H // model.blocks[0], 
                                                            model.blocks[1], 
                                                            W // model.blocks[1], 
                                                            -1, C))
        outs = outs.permute((0, 2, 1, 3, 4, 5)).reshape((model.blocks[0], 
                                                                model.blocks[1], 
                                                                -1, C))        
    else:
        outs = outs.permute((0, 2, 3, 1)).reshape((-1, C))
    model.compute_stats(outs)
    del outs
    torch.cuda.empty_cache()

    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(t))
    print("Saving model...")
    save_name = 'xuadetect/models/{}.pth'.format(spec_name if not if_side else f'{spec_name}_side')
    dir_name = os.path.dirname(save_name)
    os.makedirs(dir_name, exist_ok=True)
    memory_bank = collections.OrderedDict()
    memory_bank['memory_bank'] =  model.memory_bank
    state_dict = {"stats": memory_bank}
    torch.save(state_dict, save_name)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Save model in {}".format(str(save_name)))

def remove_white_cols(img_cv):
        # 将图像转换为灰度图
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 获取图像列的平均亮度
        col_means = np.mean(gray, axis=0)
        threshold = np.mean(col_means)
        mask = col_means < threshold
        # 找到第一个和最后一个非白色列的索引
        first_col = np.argmax(mask)

        # 找到最后一个 True 的位置
        last_col = len(mask) - 1 - np.argmax(np.flip(mask))

        # 从原始图像中裁剪除去白色条纹的部分
        result = img_cv[:, first_col: last_col]

        return result, first_col

def TrainingProc(spec_name, cfg):
    # 分割处理图片

    img_dir = f'xuadetect/img_raw/{spec_name}'
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    save_dir = f'xuadetect/img_cut/{spec_name}/raw_sel'
    save_side_dir = f'xuadetect/img_cut/{spec_name}_side/raw_sel'
    padim_dir = f'xuadetect/img_cut/{spec_name}/padim'
    padim_side_dir = f'xuadetect/img_cut/{spec_name}_side/padim'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_side_dir):
        os.makedirs(save_side_dir)
    if not os.path.exists(padim_dir):
        os.makedirs(padim_dir)
    if not os.path.exists(padim_side_dir):
        os.makedirs(padim_side_dir)
    col_num = 3
    random_list = list(range(len(img_files)))
    random.shuffle(random_list)
    system_random = random.SystemRandom()
    for img_num, img_file in enumerate(img_files):
        img = cv2.imread(os.path.join(img_dir, img_file))
        if img.shape[1] > 2000:
            img = cv2.resize(img, (img.shape[1] // 2, int(img.shape[0] // 1.5)), interpolation=cv2.INTER_CUBIC)
        img, _ = remove_white_cols(img) 
        i = 0
        patch_length = int(img.shape[1] // col_num)
        row_num = img.shape[0] // patch_length
        for i in range(0, row_num):
            random_num = system_random.randint(0, 1)
            for j in range(0, col_num):
                if patch_length * (i + 1) < img.shape[0] - cfg["remove_pixels"]:
                    rg = img[patch_length * i: patch_length * (i + 1), patch_length * j: patch_length * (j + 1)]
                # elif img.shape[0] - patch_length * i > patch_length / 2:
                #     rg = img[img.shape[0] - patch_length: img.shape[0], patch_length * j: patch_length * (j + 1)]
                else:
                    break
                if (random_num == 0 and j == 0) or (random_num == 1 and j == 2):
                    continue
                cv2.imwrite(os.path.join(save_dir if j == 1 else save_side_dir, f'{os.path.splitext(img_file)[0]}_{j}_{i}.jpg'), rg if j < 2 else cv2.flip(rg, 1))
                if random_list[img_num] < cfg['padim_train_num']:
                    cv2.imwrite(os.path.join(padim_dir if j == 1 else padim_side_dir, f'{os.path.splitext(img_file)[0]}_{j}_{i}.jpg'), rg if j < 2 else cv2.flip(rg, 1))
                    

    process = multiprocessing.Process(target=train, args=(spec_name, False, cfg))
    process_side = multiprocessing.Process(target=train, args=(spec_name, True, cfg))
    process.start()
    process_side.start()
    process.join()
    process_side.join()

    if not os.path.exists('xuadetect/loadlist'):
        os.makedirs('xuadetect/loadlist/')
    if os.path.exists(f'xuadetect/models/{spec_name}.pth') and os.path.exists(f'xuadetect/models/{spec_name}_side.pth'):
        with open(f'xuadetect/loadlist/{spec_name}', "w") as f:
            pass
    if os.path.exists(f'xuadetect/trainlist/{spec_name}'):
        os.remove(f'xuadetect/trainlist/{spec_name}')

    train_list = os.listdir('xuadetect/trainlist')                
    if len(train_list) > 0:
        TrainingProc(train_list[0], cfg)

    