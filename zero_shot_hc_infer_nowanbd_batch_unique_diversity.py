"""
This file is for concept generation in STEP 1.
We implement five kinds of methods for concept generation.
"""

# obtain the class name


import argparse
import os

import time
from copy import deepcopy

from PIL import Image
import numpy as np
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import wandb
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.concept_clip import ConceptCLIP
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from data.imagnet_prompts import *
from data.datautils import read_json_file
from concept_gen.system_prompts import SYSTEM_PROMPTS_MAPS, SYSTEM_PROMPTS_MAPS_Prmpt

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")


                     and callable(models.__dict__[name]))

import torch
import itertools
import numpy as np
from typing import List, Dict, Union, Tuple
from clip import clip

def compute_text_embeddings(texts: List[str], model, device='cuda', batch_size=32) -> torch.Tensor:
    """
    Compute text embeddings using CLIP text encoder with batching
    
    Args:
        texts: List[str] - List of text prompts
        model: ConceptCLIP - ConceptCLIP model instance
        device: str - Device to run the model on
        batch_size: int - Size of each batch
    
    Returns:
        torch.Tensor - Normalized text embeddings
    """
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # 使用CLIP的tokenize和encode_text方法
        text_tokens = clip.tokenize(batch_texts).to(device)
        with torch.no_grad():
            text_features = model.clip.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_features.append(text_features)
    
    return torch.cat(all_features, dim=0)

def compute_diversity_score(embeddings: torch.Tensor) -> float:
    """
    Compute diversity score based on cosine similarities
    
    Args:
        embeddings: Normalized text embeddings
    
    Returns:
        Diversity score (higher means more diverse)
    """
    similarities = torch.mm(embeddings, embeddings.t())
    # Remove diagonal elements (self-similarities)
    mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
    similarities = similarities[mask].view(similarities.shape[0], -1)
    # Use negative mean similarity as diversity score
    return -torch.mean(similarities).item()

def sample_from_dict(data_dict, n, batch=True, random_sample=True, mode='single', combine_op='and', num_prompts=2, 
                    clip_model=None, device='cuda', method='no', max_combinations=1000):
    """
    从字典中采样元素，支持多种采样模式。
    采样逻辑:
    如果method为no:
    
        如果mode为single，则直接从data_dict中采样n个元素。
        如果mode为multiple，则从data_dict中采样num_prompts个元素，然后组合成一个prompt。
    如果method为dpp:
        使用DPP采样方法从data_dict中采样n个元素。
    如果method为brute_force:
        使用暴力遍历方法从data_dict中采样n个元素。
    Args:
        data_dict: 要从中采样的字典, batch=True时为列表, batch=False时为字典
        n: 需要采样的元素数
        batch: 是否为批处理模式
        random_sample: 是否随机采样
        mode: 采样模式，可选值：'single'（单个prompt）, 'multiple'（多个prompt组合）
        combine_op: 当mode='multiple'时的组合方式，可选值：'or', 'and'
        num_prompts: 当mode='multiple'时要组合的prompt数量
        clip_model: CLIP模型实例，仅在使用diversity采样时需要
        device: 运行设备
        method: 采样方法，可选值：'no'（普通随机采样）, 'dpp'（DPP采样）, 'brute_force'（暴力遍历）
        max_combinations: 最大组合数量限制
    Returns:
        采样的键值对列表
    """
    def sample_dpp(candidates: List[str], n: int, clip_model, device='cuda', batch_size=64) -> List[str]:
        """
        使用DPP进行多样性采样，支持批处理
        
        Args:
            candidates: 候选prompt列表
            n: 需要采样的数量
            clip_model: CLIP模型
            device: 运行设备
            batch_size: 批处理大小
        
        Returns:
            采样得到的prompt列表
        """
        # 分批计算所有候选项的embeddings
        embeddings = compute_text_embeddings(candidates, clip_model, device, batch_size)
        
        # 分批计算相似度核矩阵
        L = torch.zeros((len(candidates), len(candidates)), device=device)
        for i in range(0, len(candidates), batch_size):
            i_end = min(i + batch_size, len(candidates))
            for j in range(0, len(candidates), batch_size):
                j_end = min(j + batch_size, len(candidates))
                L[i:i_end, j:j_end] = torch.mm(embeddings[i:i_end], embeddings[j:j_end].t())
        
        # DPP采样
        N = len(candidates)
        selected = []
        remaining = list(range(N))
        
        for _ in range(min(n, N)):
            if not remaining:
                break
                
            # 计算每个剩余项的边缘概率
            if not selected:
                # 第一个项的边缘概率就是对角线元素
                probs = torch.diagonal(L)[remaining]
            else:
                # 分批计算条件概率
                selected_embeddings = embeddings[selected]
                remaining_embeddings = embeddings[remaining]
                
                # 分批计算条件核矩阵
                probs = torch.zeros(len(remaining), device=device)
                for i in range(0, len(remaining), batch_size):
                    i_end = min(i + batch_size, len(remaining))
                    batch_remaining = remaining[i:i_end]
                    batch_embeddings = embeddings[batch_remaining]
                    
                    C = torch.mm(batch_embeddings, selected_embeddings.t())
                    A = torch.mm(selected_embeddings, selected_embeddings.t())
                    A_inv = torch.inverse(A + torch.eye(len(selected), device=device) * 1e-5)
                    
                    probs[i:i_end] = torch.diagonal(L)[batch_remaining] - torch.sum(C * torch.mm(C, A_inv), dim=1)
            
            # 选择概率最大的项
            idx = remaining[torch.argmax(probs).item()]
            selected.append(idx)
            remaining.remove(idx)
        
        return [candidates[i] for i in selected]

    def sample_brute_force(candidates: List[str], n: int, clip_model, device='cuda', batch_size=32) -> List[str]:
        """
        使用暴力遍历方法找到最大化diversity的组合，支持批处理
        
        Args:
            candidates: 候选prompt列表
            n: 需要采样的数量
            clip_model: CLIP模型
            device: 运行设备
            batch_size: 批处理大小
        
        Returns:
            采样得到的prompt列表
        """
        # 如果组合数量过大，先随机采样一部分候选项
        if len(candidates) > 100:
            candidates = random.sample(candidates, 100)
        
        # 生成所有可能的n个prompt的组合
        all_combinations = list(itertools.combinations(candidates, min(n, len(candidates))))
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
        
        best_score = float('-inf')
        best_combination = None
        
        # 分批处理组合
        for i in range(0, len(all_combinations), batch_size):
            batch_combinations = all_combinations[i:i + batch_size]
            # 将每个组合展平成一个列表
            batch_prompts = [prompt for combination in batch_combinations for prompt in combination]
            
            # 计算这一批组合的embeddings
            embeddings = compute_text_embeddings(batch_prompts, clip_model, device, batch_size)
            
            # 重新组织embeddings，使每个组合的embeddings在一起
            n_prompts = min(n, len(candidates))
            batch_embeddings = embeddings.view(len(batch_combinations), n_prompts, -1)
            
            # 计算每个组合的diversity分数
            for j, combination_embeddings in enumerate(batch_embeddings):
                score = compute_diversity_score(combination_embeddings)
                if score > best_score:
                    best_score = score
                    best_combination = batch_combinations[j]
        
        return list(best_combination) if best_combination else candidates[:n]

    def get_single_sample():
        if batch:
            if random_sample:
                return random.choices(data_dict, k=1)[0]
            else:
                return data_dict[0]
        else:
            return random.choice(list(data_dict.items()))[1]

    if method != 'no' and clip_model is None:
        raise ValueError("CLIP model is required for diversity sampling")

    if method == 'no':
        # 使用原有的随机采样方式
        if mode == 'single':
            # 原有的单个prompt采样逻辑
            if batch:
                if random_sample:
                    sampled_items = random.choices(data_dict, k=n)
                else:
                    sampled_items = data_dict[:n]
            else:
                sampled_items = random.choices(list(data_dict.items()), k=n)

            # 将采样的键值对列表转换为字典
            if batch:
                sampled_dict = {i: value for i, value in enumerate(sampled_items)}
            else:
                sampled_dict = {i: value for i, (_, value) in enumerate(sampled_items)}
        
        elif mode == 'multiple':
            sampled_dict = {}
            for i in range(n):
                # 采样多个prompt
                prompts = []
                for _ in range(num_prompts):
                    prompts.append(get_single_sample())
                
                combined_prompt = f' {combine_op} '.join(prompts)
                sampled_dict[i] = combined_prompt

    else:
        # 使用多样性采样方式
        if mode == 'single':
            # 获取候选prompt列表
            if batch:
                candidates = data_dict
            else:
                candidates = [v for _, v in data_dict.items()]
            
            # 根据指定方法进行多样性采样
            if method == 'dpp':
                sampled_items = sample_dpp(candidates, n, clip_model, device)
            else:  # brute_force
                sampled_items = sample_brute_force(candidates, n, clip_model, device)
            
            # 将采样结果转换为字典
            sampled_dict = {i: value for i, value in enumerate(sampled_items)}
        
        elif mode == 'multiple':
            # 生成所有可能的prompt组合
            if batch:
                base_candidates = data_dict
            else:
                base_candidates = [v for _, v in data_dict.items()]
            
            # 生成组合prompt
            all_combinations = []
            for _ in range(min(max_combinations, len(base_candidates) ** num_prompts)):
                prompts = random.sample(base_candidates, num_prompts)
                combined_prompt = f' {combine_op} '.join(prompts)
                all_combinations.append(combined_prompt)
            
            # 使用指定方法进行多样性采样
            if method == 'dpp':
                sampled_items = sample_dpp(all_combinations, n, clip_model, device)
            else:  # brute_force
                sampled_items = sample_brute_force(all_combinations, n, clip_model, device)
            
            # 将采样结果转换为字典
            sampled_dict = {i: value for i, value in enumerate(sampled_items)}


    return sampled_dict


def gaussian_kernel(mu, bandwidth, datapoints):
    dist = torch.norm(datapoints - mu, dim=-1, p=2)
    density = torch.exp(-dist ** 2 / (2 * bandwidth ** 2))
    return density


def solve_mta(model, inputs, args):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            image_features, text_features, logit_scale = model(inputs, features=True)
    logits = image_features @ text_features.t() * logit_scale

    lambda_y = args.lambda_y
    lambda_q = args.lambda_q
    max_iter = 5
    temperature = 1

    batch_size = image_features.shape[0]

    # bandwidth
    dist = torch.cdist(image_features, image_features)  # N,N
    sorted_dist, _ = torch.sort(dist, dim=1)
    k = int(0.3 * (image_features.shape[0] - 1))
    selected_distances = sorted_dist[:, 1:k + 1] ** 2  # exclude the distance to the point itself
    mean_distance = torch.mean(selected_distances, dim=1)  # N
    bandwidth = torch.sqrt(0.5 * mean_distance)

    # Affinity matrix based on logits
    affinity_matrix = (logits / temperature).softmax(1) @ (logits / temperature).softmax(1).t()

    # Inlierness scores initialization: uniform
    y = torch.ones(batch_size, device=image_features.device) / batch_size

    # Mode initialization: original image embedding
    mode_init = image_features[0]
    mode = mode_init

    convergence = False
    th = 1e-6
    iter = 0

    while not convergence:

        ###################
        # Inlierness step #
        ###################

        density = gaussian_kernel(mode, bandwidth, image_features)

        convergence_inlierness = False
        i = 0
        while not convergence_inlierness:
            i += 1
            old_y = y
            weighted_affinity = affinity_matrix * y.unsqueeze(0)
            y = F.softmax(1 / lambda_y * (density + lambda_q * torch.sum(weighted_affinity, dim=1)), dim=-1)

            if torch.norm(old_y - y) < th or i >= max_iter:
                convergence_inlierness = True

        #############
        # Mode step #
        #############

        convergence_mode = False
        i = 0
        while not convergence_mode:
            i += 1
            old_mode = mode
            density = gaussian_kernel(mode, bandwidth, image_features)
            weighted_density = density * y
            mode = torch.sum(weighted_density.unsqueeze(1) * image_features, dim=0) / torch.sum(weighted_density)
            mode /= mode.norm(p=2, dim=-1)

            if torch.norm(old_mode - mode) < th or i >= max_iter:
                convergence_mode = True

        iter += 1
        if iter >= max_iter:
            convergence = True

    output = mode.unsqueeze(0) @ text_features.t() * logit_scale
    return output


def concept_entropy_minimize_mta(model, images, args):
    with torch.no_grad():
        image_features, text_features, logit_scale = model.forward_concept_feature(images)  # (N, K, C) N is the number of views
    # print(outputs.requires_grad)
        text_features = torch.mean(text_features, dim=1)
        text_features = text_features/text_features.norm(-1, keepdim=True)
        logits = image_features @ text_features.t() * logit_scale

        lambda_y = args.lambda_y
        lambda_q = args.lambda_q
        max_iter = 5
        temperature = 1

        batch_size = image_features.shape[0]

        # bandwidth
        dist = torch.cdist(image_features, image_features)  # N,N
        sorted_dist, _ = torch.sort(dist, dim=1)
        k = int(0.3 * (image_features.shape[0] - 1))
        selected_distances = sorted_dist[:, 1:k + 1] ** 2  # exclude the distance to the point itself
        mean_distance = torch.mean(selected_distances, dim=1)  # N
        bandwidth = torch.sqrt(0.5 * mean_distance)

        # Affinity matrix based on logits
        affinity_matrix = (logits / temperature).softmax(1) @ (logits / temperature).softmax(1).t()

        # Inlierness scores initialization: uniform
        y = torch.ones(batch_size, device=image_features.device) / batch_size

        # Mode initialization: original image embedding
        mode_init = image_features[0]
        mode = mode_init

        convergence = False
        th = 1e-6
        iter = 0

        while not convergence:

            ###################
            # Inlierness step #
            ###################

            density = gaussian_kernel(mode, bandwidth, image_features)

            convergence_inlierness = False
            i = 0
            while not convergence_inlierness:
                i += 1
                old_y = y
                weighted_affinity = affinity_matrix * y.unsqueeze(0)
                y = F.softmax(1 / lambda_y * (density + lambda_q * torch.sum(weighted_affinity, dim=1)), dim=-1)

                if torch.norm(old_y - y) < th or i >= max_iter:
                    convergence_inlierness = True

            #############
            # Mode step #
            #############

            convergence_mode = False
            i = 0
            while not convergence_mode:
                i += 1
                old_mode = mode
                density = gaussian_kernel(mode, bandwidth, image_features)
                weighted_density = density * y
                mode = torch.sum(weighted_density.unsqueeze(1) * image_features, dim=0) / torch.sum(weighted_density)
                mode /= mode.norm(p=2, dim=-1)

                if torch.norm(old_mode - mode) < th or i >= max_iter:
                    convergence_mode = True

            iter += 1
            if iter >= max_iter:
                convergence = True

        output = mode.unsqueeze(0) @ text_features.t() * logit_scale
        return output



def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def concept_entropy_minimize(model, images, args, scaler):
    with torch.no_grad():
        outputs = model.forward_concept(images)  # (N, K, C) N is the number of views
    # print(outputs.requires_grad)
    N, K, C = outputs.shape
    w = torch.ones(K, C)/C
    w = nn.Parameter(w.cuda())  # 第一步：初始化为全1张量


    optimizer = torch.optim.AdamW([w], args.lr)
    selected_idx = None
    old_loss = torch.inf
    thres = 1e-6
    for j in range(args.tta_steps):
        with torch.amp.autocast(device_type="cuda"):
            weights = F.softmax(w*args.tau, dim=-1)
            weights = weights.unsqueeze(0)
            output = (outputs * weights).sum(dim=-1) * (model.logit_scale.exp())
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)
            loss = avg_entropy(output)
            new_loss = loss.item()
            if new_loss >old_loss or old_loss-new_loss<thres:
                break
            else:
                old_loss = new_loss
            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()
    with torch.no_grad():
        weights = F.softmax(w, dim=-1)
        weights = weights.unsqueeze(0)
        output = (outputs * weights).sum(dim=-1)  # (N,K)
        output = output[0] # 
    return output

def confident_views_ensemble(model, images, args, scaler):
    """
    直接使用confident的视图做ensemble，不使用concept权重
    
    Args:
        model: 模型
        images: 输入图像
        args: 参数
        scaler: 梯度缩放器
    
    Returns:
        output: 集成后的预测结果
    """
    with torch.no_grad():
        # 获取每个视图的预测结果
        outputs = model.forward_concept(images)  # (N, K, C) N是视图数量，K是类别数，C是概念数
        
        # 对每个视图的所有concept取平均，得到每个类别的预测分数
        logits = outputs.mean(dim=-1) * model.logit_scale.exp()  # (N, K)
        
        # 使用select_confident_samples选择confident的视图
        selected_logits, _ = select_confident_samples(logits, args.selection_p)
        
        # 对选中的视图做简单平均
        output = selected_logits.mean(dim=0)  # (K,)
        
    return output


# def multi_view_concept_avg_select_views(model, images, args, scaler, concept_method='concept_avg'):
#     """
#     Method 1: Apply concept aggregation (concept_avg or concept_mad_noise) to get B×K,
#     then select views with lowest entropy and average them.
    
#     Args:
#         model: ConceptCLIP model
#         images: Input images (B views)
#         args: Arguments containing selection_p, tau, lambda_threshold
#         scaler: Gradient scaler (not used but kept for consistency)
#         concept_method: 'concept_avg' or 'concept_mad_noise'
    
#     Returns:
#         output: Final similarity scores (K,)
#     """
#     with torch.no_grad():
#         # Get concept-level similarities: (B, K, C)
#         outputs = model.forward_concept(images)  # (B, K, C)
#         B, K, C = outputs.shape
        
#         # Apply concept aggregation to get (B, K)
#         if concept_method == 'concept_avg':
#             # Use concept_avg: average over concept dimension
#             logits = outputs.mean(dim=-1) * model.logit_scale.exp()  # (B, K)
#         elif concept_method == 'concept_mad_noise':
#             # Use concept_mad_noise logic on concept dimension
#             tau = getattr(args, 'tau', 1.0)
#             lambda_threshold = getattr(args, 'lambda_threshold', 2.5)
#             similarities = outputs * model.logit_scale.exp() / tau  # (B, K, C)
            
#             # Apply MAD noise algorithm on concept dimension
#             median = similarities.median(dim=-1, keepdim=True)[0]  # (B, K, 1)
#             mad = (similarities - median).abs().median(dim=-1, keepdim=True)[0] + 1e-6  # (B, K, 1)
            
#             outliers = (similarities - median).abs() > lambda_threshold * mad
#             rhohat = outliers.float().mean(dim=-1, keepdim=True)  # (B, K, 1)
            
#             # Compute kappa and weights
#             kappa = torch.log((1 - rhohat) / rhohat).clamp(0.1, 4.0)
#             z = -kappa * (similarities - median) / mad
#             weights = 1 / (1 + torch.exp(z))
            
#             # Weighted average over concept dimension
#             weighted_sims = weights * similarities
#             logits = weighted_sims.sum(dim=-1) / weights.sum(dim=-1)  # (B, K)
#         else:
#             raise ValueError(f"Unknown concept_method: {concept_method}")
        
#         # Select views with lowest entropy
#         selected_logits, _ = select_confident_samples(logits, args.selection_p)
        
#         # Average selected views
#         output = selected_logits.mean(dim=0)  # (K,)
        
#     return output


def multi_view_concept_avg_select_views(model, images, args, scaler, concept_method='concept_avg'):
    """
    Method 1: Apply concept aggregation (concept_avg or concept_mad_noise) to get B×K,
    then select views with lowest entropy and average them.
    
    Args:
        model: ConceptCLIP model
        images: Input images (B views)
        args: Arguments containing selection_p, tau, lambda_threshold
        scaler: Gradient scaler (not used but kept for consistency)
        concept_method: 'concept_avg' or 'concept_mad_noise'
    
    Returns:
        output: Final similarity scores (K,)
    """
    with torch.no_grad():
        # Get concept-level similarities: (B, K, C)
        outputs = model.forward_concept(images)  # (B, K, C)
        B, K, C = outputs.shape
        
        # Apply concept aggregation to get (B, K)
        if concept_method == 'concept_avg':
            # Use concept_avg: average over concept dimension
            logits = outputs.mean(dim=-1) * model.logit_scale.exp()  # (B, K)
        elif concept_method == 'concept_mad_noise':
            # Use concept_mad_noise logic on concept dimension
            tau = getattr(args, 'tau', 1.0)
            lambda_threshold = getattr(args, 'lambda_threshold', 2.5)
            similarities = outputs * model.logit_scale.exp() / tau  # (B, K, C)
            
            # Apply MAD noise algorithm on concept dimension
            median = similarities.median(dim=-1, keepdim=True)[0]  # (B, K, 1)
            mad = (similarities - median).abs().median(dim=-1, keepdim=True)[0] + 1e-6  # (B, K, 1)
            
            outliers = (similarities - median).abs() > lambda_threshold * mad
            rhohat = outliers.float().mean(dim=-1, keepdim=True)  # (B, K, 1)
            
            # Compute kappa and weights
            kappa = torch.log((1 - rhohat) / rhohat).clamp(0.1, 4.0)
            z = -kappa * (similarities - median) / mad
            weights = 1 / (1 + torch.exp(z))
            
            # Weighted average over concept dimension
            weighted_sims = weights * similarities
            logits = weighted_sims.sum(dim=-1) / weights.sum(dim=-1)  # (B, K)
        else:
            raise ValueError(f"Unknown concept_method: {concept_method}")
        
        # Select views with lowest entropy
        selected_logits, _ = select_confident_samples(logits, args.selection_p)
        
        # Average selected views
        output = selected_logits.mean(dim=0)  # (K,)
        
    return output


def multi_view_concept_robust_aggregate(model, images, args, scaler, concept_method='concept_avg'):
    """
    Method 2: Apply concept aggregation (concept_avg or concept_mad_noise) to get B×K,
    then use robust algorithm (like concept_mad_noise) to aggregate views.
    
    Args:
        model: ConceptCLIP model
        images: Input images (B views)
        args: Arguments containing tau, lambda_threshold
        scaler: Gradient scaler (not used but kept for consistency)
        concept_method: 'concept_avg' or 'concept_mad_noise'
    
    Returns:
        output: Final similarity scores (K,)
    """
    with torch.no_grad():
        # Get concept-level similarities: (B, K, C)
        outputs = model.forward_concept(images)  # (B, K, C)
        B, K, C = outputs.shape
        
        # Apply concept aggregation to get (B, K)
        if concept_method == 'concept_avg':
            # Use concept_avg: average over concept dimension
            logits = outputs.mean(dim=-1) * model.logit_scale.exp()  # (B, K)
        elif concept_method == 'concept_mad_noise':
            # Use concept_mad_noise logic on concept dimension
            tau = getattr(args, 'tau', 1.0)
            lambda_threshold = getattr(args, 'lambda_threshold', 2.5)
            similarities = outputs * model.logit_scale.exp() / tau  # (B, K, C)
            
            # Apply MAD noise algorithm on concept dimension
            median = similarities.median(dim=-1, keepdim=True)[0]  # (B, K, 1)
            mad = (similarities - median).abs().median(dim=-1, keepdim=True)[0] + 1e-6  # (B, K, 1)
            
            outliers = (similarities - median).abs() > lambda_threshold * mad
            rhohat = outliers.float().mean(dim=-1, keepdim=True)  # (B, K, 1)
            
            # Compute kappa and weights
            kappa = torch.log((1 - rhohat) / rhohat).clamp(0.1, 4.0)
            z = -kappa * (similarities - median) / mad
            weights = 1 / (1 + torch.exp(z))
            
            # Weighted average over concept dimension
            weighted_sims = weights * similarities
            logits = weighted_sims.sum(dim=-1) / weights.sum(dim=-1)  # (B, K)
        else:
            raise ValueError(f"Unknown concept_method: {concept_method}")
        
        # Apply robust aggregation on view dimension (B, K) -> (K,)
        # Transpose to (K, B) for easier processing
        logits_t = logits.t()  # (K, B)
        
        # Apply MAD noise algorithm on view dimension
        # For concept_mad_noise, logits already includes /tau from concept aggregation
        # For concept_avg, logits is at original scale (logit_scale * similarity)
        # We apply tau scaling for view dimension robust aggregation
        tau = getattr(args, 'tau', 1.0)
        lambda_threshold = getattr(args, 'lambda_threshold', 2.5)
        
        # Normalize logits: concept_mad_noise already has /tau, concept_avg needs /tau
        if concept_method == 'concept_mad_noise':
            # logits from concept_mad_noise is already scaled by /tau, use directly
            similarities_view = logits_t  # (K, B)
        else:
            # logits from concept_avg is at original scale, apply tau
            similarities_view = logits_t / tau  # (K, B)
        median_view = similarities_view.median(dim=-1, keepdim=True)[0]  # (K, 1)
        mad_view = (similarities_view - median_view).abs().median(dim=-1, keepdim=True)[0] + 1e-6  # (K, 1)
        
        outliers_view = (similarities_view - median_view).abs() > lambda_threshold * mad_view
        rhohat_view = outliers_view.float().mean(dim=-1, keepdim=True)  # (K, 1)
        
        # Compute kappa and weights
        kappa_view = torch.log((1 - rhohat_view) / rhohat_view).clamp(0.1, 4.0)
        z_view = -kappa_view * (similarities_view - median_view) / mad_view
        weights_view = 1 / (1 + torch.exp(z_view))
        
        # Weighted average over view dimension
        weighted_sims_view = weights_view * similarities_view
        output = weighted_sims_view.sum(dim=-1) / weights_view.sum(dim=-1)  # (K,)
        
        # Scale back
        output = output * tau
        
    return output


def multi_view_select_then_concept_robust(model, images, args, scaler):
    """
    Method 3: First select views, then average to get K×C similarities,
    then apply concept_mad_noise robust algorithm to aggregate concepts.
    
    Args:
        model: ConceptCLIP model
        images: Input images (B views)
        args: Arguments containing selection_p, tau, lambda_threshold
        scaler: Gradient scaler (not used but kept for consistency)
    
    Returns:
        output: Final similarity scores (K,)
    """
    with torch.no_grad():
        # Get concept-level similarities: (B, K, C)
        outputs = model.forward_concept(images)  # (B, K, C)
        B, K, C = outputs.shape
        
        # Compute logits for each view to select confident views
        logits_per_view = outputs.mean(dim=-1) * model.logit_scale.exp()  # (B, K)
        
        # Select views with lowest entropy
        selected_logits, selected_idx = select_confident_samples(logits_per_view, args.selection_p)
        
        # Average selected views to get (K, C) similarities
        # outputs[selected_idx] is (selected_B, K, C)
        # We want to average over view dimension to get (K, C)
        selected_outputs = outputs[selected_idx]  # (selected_B, K, C)
        avg_concept_similarities = selected_outputs.mean(dim=0) * model.logit_scale.exp()  # (K, C)
        
        # Apply concept_mad_noise robust algorithm on concept dimension
        tau = getattr(args, 'tau', 1.0)
        lambda_threshold = getattr(args, 'lambda_threshold', 2.5)
        
        similarities = avg_concept_similarities / tau  # (K, C)
        median = similarities.median(dim=-1, keepdim=True)[0]  # (K, 1)
        mad = (similarities - median).abs().median(dim=-1, keepdim=True)[0] + 1e-6  # (K, 1)
        
        outliers = (similarities - median).abs() > lambda_threshold * mad
        rhohat = outliers.float().mean(dim=-1, keepdim=True)  # (K, 1)
        
        # Compute kappa and weights
        kappa = torch.log((1 - rhohat) / rhohat).clamp(0.1, 4.0)
        z = -kappa * (similarities - median) / mad
        weights = 1 / (1 + torch.exp(z))
        
        # Weighted average over concept dimension
        weighted_sims = weights * similarities
        output = weighted_sims.sum(dim=-1) / weights.sum(dim=-1)  # (K,)
        
        # Scale back
        output = output * tau
        
    return output


def concept_entropy_minimize_kl(model, images, args, scaler):
    with torch.no_grad():
        outputs = model.forward_concept(images)  # (N, K, C) N是视图数量，K是类别数，C是概念数
    N, K, C = outputs.shape
    w = torch.ones(K, C)/C
    w = nn.Parameter(w.cuda())
    
    optimizer = torch.optim.AdamW([w], args.lr)
    selected_idx = None
    old_loss = torch.inf
    thres = 1e-6

    for j in range(args.tta_steps):
        with torch.amp.autocast(device_type="cuda"):
            weights = F.softmax(w*args.tau, dim=-1)
            weights = weights.unsqueeze(0)
            
            # 计算每个类别每个concept在不同视图上的分布
            concept_logits = outputs  # (N, K, C)
            # print(concept_logits.shape)
            
            # 对每个样本的每个(K,C)对计算其在不同视图上的分布
            
            # 对样本维度求平均
            avg_concept_logits = concept_logits.mean(dim=0) * (model.logit_scale.exp()) # (K, C)
            
            # 计算每个(K,C)对的熵
            concept_probs = F.softmax(avg_concept_logits, dim=0)  # 在类别维度做softmax
            concept_entropy = -(concept_probs * torch.log(concept_probs + 1e-10))  # (K, C)
            # print(concept_entropy)
            
            # 计算每个类别中，concept权重和其熵的差异
            # 我们希望权重的分布和熵的分布尽量接近
            # 先将entropy归一化到[0,1]区间
            normalized_entropy = F.softmax(concept_entropy, dim=-1)  # 在concept维度上归一化 
            print(normalized_entropy.shape)
            # 计算权重分布和熵分布的KL散度
            stability_loss = F.kl_div(weights.squeeze(0).log(), normalized_entropy, reduction='batchmean')
            
            # 计算原始的熵loss
            output = (outputs * weights).sum(dim=-1) * (model.logit_scale.exp())
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)
            entropy_loss = avg_entropy(output)
            
            # 组合两个loss，注意这里stability_loss已经是负的了，所以用加号
            loss = entropy_loss + 0.5* stability_loss
            
            new_loss = loss.item()
            if new_loss > old_loss or old_loss-new_loss < thres:
                break
            else:
                old_loss = new_loss
                # print(f"Total Loss: {loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}, Stability Loss: {stability_loss.item():.4f}")
                # print(weights)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    with torch.no_grad():
        weights = F.softmax(w, dim=-1)
        weights = weights.unsqueeze(0)
        output = (outputs * weights).sum(dim=-1)
        output = output[0]
    return output

def concept_entropy_minimize_embedding(model, images, args, scaler):
    with torch.no_grad():
        image_features, prompts, logit_scale = model.forward_concept_feature(images)  # (N, K, C) N is the number of views
    # print(outputs.requires_grad)
    K, C, _ = prompts.shape
    w = nn.Parameter(torch.ones(K, C).cuda())  # 第一步：初始化为全1张量
    nn.init.constant_(w, 1.0 / C)
    optimizer = torch.optim.AdamW([w], args.lr)
    selected_idx = None
    old_loss = torch.inf
    thres = 1e-6
    for j in range(args.tta_steps):
        with torch.amp.autocast(device_type="cuda"):
            weights = F.softmax(w, dim=-1) # K,C
            weights = weights.unsqueeze(2)
            text_features = torch.sum(weights*prompts, dim=1) # K, dim
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            output = image_features @ text_features.t() * logit_scale# N,K
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)
            loss = avg_entropy(output)
            new_loss = loss.item()
            if new_loss >old_loss or old_loss-new_loss<thres:
                break
            else:
                old_loss = new_loss
            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()
    with torch.no_grad():
        weights = F.softmax(w, dim=-1)
        weights = weights.unsqueeze(2)
        text_features = torch.sum(weights * prompts, dim=1)  # K, dim
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        output = image_features[0] @ text_features.t() * logit_scale  # K
    return output


def compute_similarity_regularization_matrix(A, B):
    """
    计算逻辑值相似的概念应该具有相似的权重的正则化损失 (矩阵化版本)。

    Args:
    A: torch.Tensor, 大小为 (K, C) 的逻辑张量
    B: torch.Tensor, 大小为 (K, C) 的权重张量, 沿着 C 维度和为 1

    Returns:
    loss: torch.Tensor, 正则化损失
    """
    K, C = A.shape

    # 计算逻辑值差异矩阵 S_A (K, C, C)
    logic_diff = A.unsqueeze(2) - A.unsqueeze(1)  # (K, C, C)
    S_A = logic_diff.pow(2)  # (K, C, C)

    # 计算权重差异矩阵 S_B (K, C, C)
    weight_diff = B.unsqueeze(2) - B.unsqueeze(1)  # (K, C, C)
    S_B = weight_diff.pow(2)  # (K, C, C)

    # 对 S_A 和 S_B 逐元素相乘后求和，计算总损失
    loss = torch.sum(S_A * S_B)

    return loss


def main():
    args = parser.parse_args()
    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    if not args.aug:
        run = wandb.init(
            # Set the project where this run will be logged
            project=args.project,
            config={
                    "model mode": args.arch,
                        "algorithm mode": args.al_mode,
                    "concept mode": args.concept,
                    "tau": args.tau
            },
            name=args.name
        )
    else:
        run = wandb.init(
            # Set the project where this run will be logged
            project=args.project,
            config={
                "TTA Steps": args.tta_steps,
                "lr": args.lr,
                "tau": args.tau
            },
            name=args.name
        )

    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))
    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.ensemble:
        prompt_fix = imagenet_templates
    else:
        # prompt_fix = "a photo contains a {}."
        prompt_fix = "a photo of a {}."

    """
    write code to load parameters 
    """

    print("=> Model created: visual backbone {}".format(args.arch))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        model = ConceptCLIP(device="cpu", arch=args.arch, prompt_fix=prompt_fix, al_mode=args.al_mode, decompose_or_prompts=args.decompose_or_prompts)
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = ConceptCLIP(device="cuda", arch=args.arch, prompt_fix=prompt_fix, al_mode=args.al_mode, decompose_or_prompts=args.decompose_or_prompts)
        model = model.cuda(args.gpu)

    # # setup automatic mixed-precision (Amp) loss scaling
    # scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    all_runs_results = []  # 存储所有运行的结果
    
    for run_idx in range(args.num_runs):
        print(f"\n=== Running iteration {run_idx + 1}/{args.num_runs} ===")
        # 为每次运行设置不同的随机种子
        current_seed = [0, 42, 100]
        set_random_seed(current_seed[run_idx])
        print(f"Using random seed: {current_seed}")
        
        result_all = []
        sampling_times = args.sampling_times
            # iterating through eval datasets
        datasets = args.test_sets.split("/")
        results = {}

        for set_id in datasets:
            if args.aug:
                # use test time augmentation
                base_transform = transforms.Compose([
                    transforms.Resize(args.resolution, interpolation=BICUBIC),
                    transforms.CenterCrop(args.resolution)])
                preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    normalize])
                data_transform = AugMixAugmenter(base_transform, preprocess, n_views=64 - 1,
                                                 augmix=len(set_id) > 1)
                batchsize = 1
            else:
                data_transform = transforms.Compose([
                    transforms.Resize(args.resolution, interpolation=BICUBIC),
                    transforms.CenterCrop(args.resolution),
                    transforms.ToTensor(),
                    normalize,
                ])
                batchsize = args.batch_size

            print("evaluating: {}".format(set_id))
            # reset the model
            # Reset classnames of custom CLIP model
            if len(set_id) > 1:
                # fine-grained classification datasets. For imagenet and variants of imageset, we use capital letter as ID
                classnames = eval("{}_classes".format(set_id.lower()))
                concept_path = os.path.join(args.concept_dir, set_id.lower(), args.concept_type, "results.json")
                concepts = read_json_file(concept_path)
                # for ablation study of sampling  times
                if args.batch:
                    concepts["concepts"] = {
                        key: sample_from_dict(
                            concepts[key], 
                            args.sampling_times, 
                            batch=True, 
                            random_sample=args.random_sample,
                            mode=args.sample_mode, 
                            combine_op=args.combine_op, 
                            num_prompts=args.len_prompts,
                            method=args.sampling_method,
                            max_combinations=args.max_combinations,
                            clip_model=model if args.sampling_method != 'no' else None
                        ) for key in classnames
                    }
                else:
                    concepts["concepts"] = {
                        key: sample_from_dict(
                            concepts["concepts"][key], 
                            args.sampling_times, 
                            batch=False,
                            random_sample=args.random_sample,
                            method=args.sampling_method,
                            max_combinations=args.max_combinations,
                            clip_model=model if args.sampling_method != 'no' else None
                        ) for key in classnames
                    }
            else:
                assert set_id in ['A', 'R', 'K', 'V', 'I']  # different version of imagenet
                classnames_all = imagenet_classes
                classnames = []
                concept_path = os.path.join(args.concept_dir, "imagenet", args.concept_type, "results.json")
                concepts = read_json_file(concept_path)
                if set_id in ['A', 'R', 'V']:
                    label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                    if set_id == 'R':
                        for i, m in enumerate(label_mask):
                            if m:
                                classnames.append(classnames_all[i])
                    else:
                        classnames = [classnames_all[i] for i in label_mask]
                else:
                    classnames = classnames_all
                
                if args.batch:
                    concepts["concepts"] = {
                        key: sample_from_dict(
                            concepts[key], 
                            args.sampling_times, 
                            batch=True, 
                            random_sample=args.random_sample,
                            mode=args.sample_mode, 
                            combine_op=args.combine_op, 
                            num_prompts=args.len_prompts,
                            method=args.sampling_method,
                            max_combinations=args.max_combinations,
                            clip_model=model if args.sampling_method != 'no' else None
                        ) for key in classnames
                    }
                else:
                    concepts["concepts"] = {
                        key: sample_from_dict(
                            concepts["concepts"][key], 
                            args.sampling_times, 
                            batch=False,
                            random_sample=args.random_sample,
                            method=args.sampling_method,
                            max_combinations=args.max_combinations,
                            clip_model=model if args.sampling_method != 'no' else None
                        ) for key in classnames
                    }

            if set_id in ['A', 'R', 'K', 'V', 'I']:
                dataset_name = "imagenet"
                dataset_context = SYSTEM_PROMPTS_MAPS_Prmpt[dataset_name.lower()]["dataset_context"]
                # if set_id in  ['A', 'R', 'K', 'V']:
                #     dataset_context = ""
            else:
                dataset_name = set_id
                dataset_context = SYSTEM_PROMPTS_MAPS_Prmpt[dataset_name.lower()]["dataset_context"]
            model.reset_classnames(classnames, concepts, dataset_context=dataset_context)
            model.reset_prompt_embedding()

            # setup automatic mixed-precision (Amp) loss scaling
            scaler = torch.amp.GradScaler('cuda', init_scale=1000)
            # scaler = torch.amp.GradScaler('cuda')

            # has few shot implement
            val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)

            print("number of test samples: {}\n".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batchsize, shuffle=True,
                num_workers=args.workers, pin_memory=True)

            # 不同的测试算法
            if args.aug:
                print("aug mode")
                results[set_id] = test_time_adapt_eval(val_loader, model, scaler, args)
            else:
                print("no aug mode")
                results[set_id] = testset_eval(val_loader, model, args)
            del val_dataset, val_loader
            # try:
            #     print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
            # except:
            #     print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))
            # model = model.cpu()

        print("======== Result Summary ========")
        # print("params: nstep	lr	bs")
        # print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
        print(f"Sampling Times: {sampling_times}")
        print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
        for id in results.keys():
            print("{}".format(id), end="	")
        print("\n")
        all_top1 = []
        all_rhohat = []  # 新增：用于收集每个数据集的avg_rhohat
        for id in results.keys():
            # results[id] = [top1, top5, mean_rhohat]
            print("{:.2f}".format(results[id][0]), end="\t")
            all_top1.append(results[id][0].cpu().item())
            if len(results[id]) > 2 and results[id][2] is not None:
                all_rhohat.append(results[id][2])
        result_all.append(all_top1)
        print("The average top_1 score is {:.2f}".format(np.mean(np.array(all_top1))))
        if len(all_rhohat) > 0:
            avg_rhohat_allsets = np.mean(all_rhohat)
            print("所有测试集平均rhohat: {:.4f}".format(avg_rhohat_allsets))
        print("\n")
        
        # 将当前运行的结果添加到所有运行结果列表中
        all_runs_results.append(all_top1)
    
    # 打印所有运行的结果
    print("\n======== All Runs Results ========")
    all_runs_array = np.array(all_runs_results)  # shape: (num_runs, num_datasets)
    
    # 计算每个数据集的平均值和标准差
    means = np.mean(all_runs_array, axis=0)
    stds = np.std(all_runs_array, axis=0)
    
    print("\nResults for each dataset:")
    for i, dataset in enumerate(datasets):
        print(f"{dataset}:")
        print(f"  All runs: {[f'{x:.2f}' for x in all_runs_array[:, i]]}")
        print(f"  Mean: {means[i]:.2f} ± {stds[i]:.2f}")
    
    print("\nOverall statistics:")
    print(f"Mean across all datasets and runs: {np.mean(means):.2f} ± {np.std(means):.2f}")
    
    # 保存结果到JSON文件
    import json
    
    # 构建简化的结果字典
    result_dict = {
        "dataset": args.test_sets,
        "mode": args.al_mode,
        "tau": args.tau,
        "lambda": args.lambda_threshold if args.al_mode == "concept_mad_noise" else None,
        "alpha": args.alpha if args.al_mode == "concept_avg_noise" else None,
        "contamination_rate": args.contamination_rate if args.al_mode == "concept_hard_trim" else None,
        "scale_factor": args.scale_factor if args.al_mode == "concept_cauchy" else None,
        "sampling_times": args.sampling_times,
        "sample_mode": args.sample_mode,
        "combine_op": args.combine_op,
        "len_prompts": args.len_prompts,
        "max_combinations": args.max_combinations,
        "sampling_method": args.sampling_method,
        "concept_type": args.concept_type,
        "arch": args.arch,
        "ensemble": args.ensemble,
        "multi_view_method": args.multi_view_method,
        "decompose_or_prompts": args.decompose_or_prompts,
    }
    
    # 添加每个数据集的均值和标准差
    for i, dataset in enumerate(datasets):
        result_dict[f"{dataset}"] = f"{means[i]:.2f} \u00b1 {stds[i]:.2f}"
    
    # 添加总体均值和标准差
    result_dict["mean"] = f"{np.mean(means):.2f} \u00b1 {np.std(means):.2f}"
    
    # 保存平均rhohat及分集rhohat
    if len(all_rhohat) > 0:
        result_dict["mean_rhohat"] = float(avg_rhohat_allsets)
        for i, dataset in enumerate(datasets):
            if i < len(all_rhohat):
                result_dict[f"{dataset}_mean_rhohat"] = float(all_rhohat[i])
    
    # 读取现有结果（如果文件存在）
    try:
        with open(args.result_path, 'r') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = []
    
    # 添加新的结果
    all_results.append(result_dict)
    
    # 确保结果目录存在
    # 确保结果路径有效
    if not args.result_path:
        args.result_path = 'results/experiment_results.json'
    
    # 创建结果目录
    result_dir = os.path.dirname(args.result_path)
    if result_dir:  # 如果文件在当前目录，dirname会返回空字符串
        os.makedirs(result_dir, exist_ok=True)
    
    # 写入所有结果
    with open(args.result_path, 'w+') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults appended to: {args.result_path}")
    
    # wandb logging
    if wandb.run is not None:
        for i, dataset in enumerate(datasets):
            wandb.log({
                f"{dataset}_mean": means[i],
                f"{dataset}_std": stds[i]
            })
        wandb.log({
            "overall_mean": np.mean(means),
            "average_std": np.mean(stds)
        })

        # if not args.aug:
        #     print("model mode {} algorithm mode {} concept mode {} tau {}".format(args.arch, args.al_mode, args.concept, args.tau))
        #
        # else:
        #     print(
        #         "Aug mode {}  TTA Steps {} Learning rate {}".format(args.aug, args.tta_steps, args.lr))




def testset_eval(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    entropy = AverageMeter('Entropy', ':6.2f', Summary.AVERAGE)
    rhohat_list = []    # 新增：用于统计所有batch的rhohat

    # reset model and switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (images, target) in enumerate(tqdm(val_loader)):
        images = images.cuda(args.gpu, non_blocking=True)
        targets = target.cuda(args.gpu, non_blocking=True)
        if args.al_mode == 'concept_map':
            outputs = model.zero_shot_concept_map(images)
        elif args.al_mode == 'concept_avg':
            outputs = model.zero_shot_concept_avg(images, mask=args.cmask)
        elif args.al_mode == 'concept_prior_map':
            outputs = model.zero_shot_concept_map_prior(images, args.tau)
        elif args.al_mode == 'concept_mad_noise_two_side_huber':
            outputs = model.zero_shot_concept_mad_noise_two_side_huber(images)
        elif args.al_mode == 'concept_mad_noise':
            outputs = model.zero_shot_concept_mad_noise(images, tau=args.tau, lambda_threshold=args.lambda_threshold)
        elif args.al_mode == 'concept_avg_noise':
            outputs = model.zero_shot_concept_avg_noise(images, alpha=0.2)
        elif args.al_mode == 'concept_hard_trim':
            outputs = model.zero_shot_concept_hard_trim(images, contamination_rate=args.contamination_rate, tau=args.tau)
        elif args.al_mode == 'concept_median':
            outputs = model.zero_shot_concept_median(images, tau=args.tau)
        elif args.al_mode == 'concept_cauchy':
            outputs = model.zero_shot_concept_cauchy(images, tau=args.tau, scale_factor=args.scale_factor)
        else:
            outputs = model.zero_shot_inference(images)
        if args.al_mode == 'concept_mad_noise':
            if len(outputs) == 2:
                outputs, rhohat = outputs
                rhohat_list.append(rhohat.detach().cpu().numpy().reshape(-1))
        else:
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                outputs, _ = outputs
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        entropy.update(avg_entropy(outputs), images.size(0))
    # ========= 统计并返回平均rhohat =========
    mean_rhohat = None
    if len(rhohat_list) > 0:
        mean_rhohat = np.concatenate(rhohat_list).mean()
        print("数据集上平均rhohat: {:.4f}".format(mean_rhohat))
    return [top1.avg, top5.avg, mean_rhohat]


def test_time_adapt_eval(val_loader, model, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    # reset model and switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (images, target) in enumerate(tqdm(val_loader)):
        for k in range(len(images)):
            images[k] = images[k].cuda(args.gpu, non_blocking=True)  # sdfk
        image = images[0]
        target = target.cuda(args.gpu, non_blocking=True)
        images = torch.cat(images, dim=0)
        # if args.tpt:
        #     images = torch.cat(images, dim=0)
        if args.tta_steps > 0:
            # Select multi-view method based on args.multi_view_method
            if hasattr(args, 'multi_view_method') and args.multi_view_method is not None:
                if args.multi_view_method == 'method1_concept_avg':
                    output = multi_view_concept_avg_select_views(model, images, args, scaler, concept_method='concept_avg').unsqueeze(0)
                elif args.multi_view_method == 'method1_concept_mad_noise':
                    output = multi_view_concept_avg_select_views(model, images, args, scaler, concept_method='concept_mad_noise').unsqueeze(0)
                elif args.multi_view_method == 'method2_concept_avg':
                    output = multi_view_concept_robust_aggregate(model, images, args, scaler, concept_method='concept_avg').unsqueeze(0)
                elif args.multi_view_method == 'method2_concept_mad_noise':
                    output = multi_view_concept_robust_aggregate(model, images, args, scaler, concept_method='concept_mad_noise').unsqueeze(0)
                elif args.multi_view_method == 'method3':
                    output = multi_view_select_then_concept_robust(model, images, args, scaler).unsqueeze(0)
                else:
                    # Default to confident_views_ensemble if unknown method
                    output = confident_views_ensemble(model, images, args, scaler).unsqueeze(0)
            else:
                # Default behavior: use confident_views_ensemble
                output = confident_views_ensemble(model, images, args, scaler).unsqueeze(0)
            # output = concept_entropy_minimize_kl_version2(model, images, args, scaler).unsqueeze(0)
            # output = concept_entropy_minimize(model, images, args, scaler).unsqueeze(0)
            # output = concept_entropy_minimize_no_aug(model, images, args, scaler).unsqueeze(0)
            # output = concept_entropy_minimize_embedding(model, images, args, scaler).unsqueeze(0)
            # output = concept_entropy_minimize_mta(model, images, args)
        else:
            outputs = model.forward_concept(images)  # (N, K, C) N is the number of views
            std_dev = torch.std(outputs, dim=0)
            top_10_percent_count = int(0.1 * outputs.shape[2])  # C 的 10%
            # output = torch.zeros(outputs.shape[1])  #
            # 对每个类别 K 进行处理
            for k in range(outputs.shape[1]):  # 遍历每个类别 K
                # 获取类别 k 的标准差, std_dev[k] 是形状为 (C,) 的 tensor
                std_k = std_dev[k]
                top_10_percent_indices = torch.argsort(std_k)[:top_10_percent_count]
                # Step 4: 对选中的 concept 在视图维度 (N) 上进行相似度平均
                # outputs[:, k, :] 是形状为 (N, C) 的 tensor，表示类别 k 的所有视图和 concept
                # 我们只对 top_10_percent_indices 进行平均
                selected_concepts = outputs[:, k, top_10_percent_indices]  # shape = (N, top_10_percent_count)
                # 对这些选中的 conc
                # ept 在视图维度 (N) 上求平均
                output[k] = selected_concepts[0].mean()
                # output[k] = selected_concepts.mean()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.cuda(), target, topk=(1, 5))

        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return [top1.avg, top5.avg]


# "food101/flower102/caltech101/dtd/aircraft
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', default='data', help='path to dataset root', )
    parser.add_argument('--al_mode', default="concept_avg", help='path to dataset root',
                        choices=["concept_map", "concept_avg", "avg", "concept_prior_map", "concept_mad_noise_two_side_huber", "concept_mad_noise", "concept_avg_noise", "concept_hard_trim", "concept_median", "concept_cauchy"])
    parser.add_argument('--tau', default=1, type=float, help='tau for concept_mad_noise_two_side_huber')
    parser.add_argument('--alpha', default=0.2, type=float, help='alpha for concept_avg_noise')
    parser.add_argument('--lambda_threshold', default=2.5, type=float, help='lambda_threshold for concept_mad_noise')
    parser.add_argument('--contamination_rate', default=0.1, type=float, help='contamination rate for concept_hard_trim')
    parser.add_argument('--scale_factor', default=1.0, type=float, help='scale factor for concept_cauchy')
    # 采样相关参数
    parser.add_argument('--sample_mode', default='single', choices=['single', 'multiple'], help='sample mode')
    parser.add_argument('--combine_op', default='or', choices=['or', 'and', ","], help='combine operation')
    parser.add_argument('--len_prompts', default=2, type=int, help='length of prompts')
    parser.add_argument('--sampling_times', default=50, type=int, help='sampling times')
    parser.add_argument('--sampling_method', default='no', choices=['no', 'dpp', 'brute_force'], help='sampling method: no (random), dpp, or brute_force')
    parser.add_argument('--max_combinations', default=500, type=int, help='maximum number of combinations to try for diversity sampling')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for diversity sampling')
    parser.add_argument('--random_sample', default=True, type=bool, help='whether to use random sampling when method is no')
    parser.add_argument('--num_runs', default=3, type=int, help='number of runs with different random seeds')
    parser.add_argument('--decompose_or_prompts', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='When enabled (True), decompose OR-combined prompts into individual concepts, '
                             'rebuild prompts for each concept using the template, and average their embeddings. '
                             'This studies the effect of OR vs direct averaging of concepts.')
    # DTD/Cars/Aircraft/UCF101/eurosat A/R/K/V/I
    # DTD/Cars/Aircraft/UCF101/eurosat/Pets/Flower102/Food101/SUN397/Caltech101
    #Caltech101/Cars/Flower102/Food101/Pets DTD/Aircraft/UCF101/SUN397/I
    # "DTD/Cars/Aircraft/UCF101/eurosat/Pets/Flower102/Food101/SUN397/Caltech101"
    # "SUN397/Aircraft/eurosat/Cars/Food101/Pets/Flower102/Caltech101/DTD/UCF101"
    # eurosat
    parser.add_argument('--test_sets', type=str, default="food101/flower102/caltech101/dtd/aircraft",
                        help='test dataset (multiple datasets split by slash)'
                        )
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', default='ViT-B/16')  # RN50 ViT-B/16
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution') # 288x288 384 448
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N') # 增强的视图的size
    parser.add_argument('--lr', '--learning-rate', default=1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    # parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile') # 0.1
    # parser.add_argument('--tau', default=1, type=float, help='confidence selection percentile')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tta_steps', default=30, type=int, help='test-time-adapt steps')
    parser.add_argument('--aug', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--concept_dir', type=str, default="concept_gen/batchconcepts")  # concept_gen/save_concepts"
    parser.add_argument('--concept_type', type=str, default="50_sim") # 10_100_sim_TPT for imagenet 10_100_nosim
    # parser.add_argument('--concept_prior', type=str, default="10_100_sim_TPT")
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='use an ensemble of prompts')
    parser.add_argument('--batch', action='store_true', default=True,
                        help='use batch sampling')
    parser.add_argument('--concept', action='store_true', default=False,
                        help='use concept')
    parser.add_argument('--cmask', action='store_true', default=False,
                        help='mask none concept in concept_avg mode')

    parser.add_argument('--lambda_q', default=4, help='quadratic term weighting factor')
    parser.add_argument('--lambda_y', default=0.2, help='entropic term weighting factor')
    parser.add_argument('--multi_view_method', type=str, default=None,
                        choices=['method1_concept_avg', 'method1_concept_mad_noise', 
                                'method2_concept_avg', 'method2_concept_mad_noise', 'method3'],
                        help='Multi-view aggregation method when aug=True. '
                             'method1_concept_avg: concept_avg then select views; '
                             'method1_concept_mad_noise: concept_mad_noise then select views; '
                             'method2_concept_avg: concept_avg then robust aggregate views; '
                             'method2_concept_mad_noise: concept_mad_noise then robust aggregate views; '
                             'method3: select views then robust aggregate concepts')

    # parser.add_argument('--sampling_times', default=20, type=int, help='confidence selection percentile')
    #####
    parser.add_argument('--project', default="Ablationn of Sampling Times", type=str)
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--result_path', type=str, default='results/experiment_results.json',
                        help='path to save results')

    main()
