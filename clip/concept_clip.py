from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
# from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .custom_clip import TextEncoder
from data.imagnet_prompts import imagenet_classes
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets

# from concept_gen.concept_prior import cluster_concept_sb, cluster_concept_clip

# _tokenizer = _Tokenizer()

DOWNLOAD_ROOT = '/hdd2/lh/PLM'


def remove_redundant_concepts(concept_dict):
    # Step 1: Remove concepts that contain other class names
    # cleaned_concepts = {}
    cleaned_concepts = {}
    class_list = concept_dict['concepts'].keys()
    i = 0
    for class_name, concepts in concept_dict['concepts'].items():
        cleaned_concepts[class_name] = {}
        for org_index, concept in concepts.items():
            if concept is not None:
                if any(other_class.lower() in concept.lower() for other_class in class_list if
                       other_class != class_name):
                    cleaned_concepts[class_name][org_index] = None
                    # print(concept)
                    i += 1
                else:
                    cleaned_concepts[class_name][org_index] = concept
            else:
                cleaned_concepts[class_name][org_index] = concept

    print(i)
    return cleaned_concepts


class ConceptCLIP(nn.Module):
    def __init__(self, device, arch="ViT-L/14", prompt_fix="a photo of a {}.", al_mode=None, decompose_or_prompts=False):
        super().__init__()
        self.clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.logit_scale = self.clip.logit_scale.data
        self.prompt_fix = prompt_fix
        self.device = device
        self.al_mode = al_mode
        self.dtype=self.clip.dtype
        self.decompose_or_prompts = decompose_or_prompts

    def reset_classnames(self, classnames, concepts, dataset_context=""):
        self.classnames = classnames
        self.concepts_dict = concepts
        self.dataset_context = dataset_context
        self.num_class = len(self.classnames)


    def reset_prompt_embedding(self):
        if self.al_mode in ["concept_map", "concept_avg", "concept_prior_map", "concept_mad_noise_two_side_huber", "concept_mad_noise", "concept_avg_noise", "concept_hard_trim", "concept_median", "concept_cauchy"]:
            self.combined_list, self.num_concepts, self.concept_mask_tensor = self.zero_shot_prompt_embedding_concept__init()
        # elif self.al_mode in ["concept_prior_map"]:
        #     self.combined_list, self.num_concepts, self.concept_mask_tensor = self.zero_shot_prompt_embedding_concept__init()
        else:
            self.combined_list = self.zero_shot_prompt_embedding_init()

        self.prompts = self.combined_list.to(self.device)  # K, C, Dim
        self.prompts = self.prompts / self.prompts.norm(dim=-1, keepdim=True)


    def zero_shot_prompt_embedding_concept__init(self):
        concept_set = self.concepts_dict["concepts"]
        combined_list = []
        if isinstance(self.prompt_fix, str):
            self.prompt_fix = [self.prompt_fix]
        for template in self.prompt_fix:
            prefix_p = []
            for class_name in self.classnames:
                prefix_p_c = []
                c_class = concept_set[class_name]
                for idc, v in c_class.items():  # key: sampling_times, value: concept
                    # template_ = template.format(class_name)  # "a photo of "
                    if "satellite" in self.dataset_context:
                        template_ = f"a {self.dataset_context} photo of {class_name}."
                    else:
                        template_ = f"a photo of a {class_name}."
                    if v is not None:
                        # Check if we need to decompose OR prompts
                        if self.decompose_or_prompts and " or " in v:
                            # Split the combined prompt by " or "
                            concepts = [c.strip() for c in v.split(" or ")]
                            # Build individual prompts for each concept
                            concept_prompts = []
                            for concept in concepts:
                                concept_template = template_[:-1] + " with " + concept + template_[-1]
                                concept_prompts.append(concept_template)
                            # Tokenize and encode all concept prompts
                            concept_prompts_tokens = tokenize(concept_prompts).to(self.device)
                            with torch.no_grad():
                                concept_embeddings = self.clip.encode_text(concept_prompts_tokens)
                                concept_embeddings = concept_embeddings / concept_embeddings.norm(dim=-1, keepdim=True)
                            # Average the embeddings of all concepts
                            avg_embedding = concept_embeddings.mean(dim=0)
                            prefix_p_c.append(avg_embedding.cpu())
                        else:
                            template_ = template_[:-1] + " with " + v + template_[-1]
                            prefix_p_c.append(template_)
                    else:
                        prefix_p_c.append(template_)
                
                # Process prompts that are strings (not yet embedded)
                prompts_to_tokenize = []
                prompt_indices = []
                for idx, prompt in enumerate(prefix_p_c):
                    if isinstance(prompt, str):
                        prompts_to_tokenize.append(prompt)
                        prompt_indices.append(idx)
                
                if prompts_to_tokenize:
                    prompts_tokens = tokenize(prompts_to_tokenize).to(self.device)
                    with torch.no_grad():
                        prompts_embeddings = self.clip.encode_text(prompts_tokens)
                        prompts_embeddings = prompts_embeddings / prompts_embeddings.norm(dim=-1, keepdim=True)
                    
                    # Replace string prompts with embeddings
                    for i, idx in enumerate(prompt_indices):
                        prefix_p_c[idx] = prompts_embeddings[i].cpu()
                
                # Stack all embeddings
                prefix_p_c = torch.stack(prefix_p_c, dim=0)
                prefix_p.append(prefix_p_c)
            prefix_p = torch.stack(prefix_p, dim=0)  # num of classes, num of concepts
            combined_list.append(prefix_p)  # num of prompts, num of classes, num of concepts

        # mask matrix is independent of different prompts
        mask_matirx = []
        for class_name in self.classnames:
            mask_prefix_p_c = []
            c_class = concept_set[class_name]
            for idc, v in c_class.items():  # key: sampling_times, value: concept
                if v is not None:
                    mask_prefix_p_c.append(1)
                else:
                    mask_prefix_p_c.append(0)
            mask_matirx.append(mask_prefix_p_c)
        mask_matirx = torch.tensor(mask_matirx)

        prompts = torch.stack(combined_list, dim=0)
        prompts = torch.mean(prompts, dim=0)  # num of classes*num of concepts, dim
        prompts /= prompts.norm(dim=-1, keepdim=True)
        #  combined_list = self.zero_shot_prompt_embedding_init()
        return prompts.cpu(), len(concept_set[self.classnames[0]]), mask_matirx

    def zero_shot_prompt_embedding_init(self):
        combined_list = []
        if isinstance(self.prompt_fix, str):
            self.prompt_fix = [self.prompt_fix]
        for template in self.prompt_fix:
            prefix_p = []
            for class_name in self.classnames:
                template_ = template.format(class_name)
                prefix_p.append(template_)
            prefix_p = tokenize(prefix_p).to(self.device)
            with torch.no_grad():
                prefix_p = self.clip.encode_text(prefix_p) # 
                prefix_p = prefix_p / prefix_p.norm(dim=-1, keepdim=True) # number of classses, dim
                combined_list.append(prefix_p.cpu()) 
        combined_list = torch.stack(combined_list, dim=0)
        return combined_list

    def zero_shot_inference(self, image, label=None):
        with torch.no_grad():
            prompts = self.combined_list.to(self.device)
            prompts = torch.mean(prompts, dim=0)  # num_classes, dim
            prompts = prompts / prompts.norm(dim=-1, keepdim=True)  # [397, 100, 1024]
            logit_scale = self.logit_scale.exp()
            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 64, 1024
            logits = []
            for img_i in image_features:
                l_i = logit_scale * img_i @ prompts.t()  # (num of classes)
                logits.append(l_i)
            logits = torch.stack(logits, dim=0)  # batchsize, lan_p_set, num of clsses
        return logits

    def zero_shot_concept_map(self, image, label=None):
        with torch.no_grad():
            # 1000,100,758
            prompts = self.combined_list.to(self.device)
            logit_scale = self.logit_scale.exp()
            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 64, 1024
            logits = []
            for img_i in image_features:
                l_i = logit_scale * prompts @ img_i  # num of classes, num of concepts * (self.concept_mask_tensor.to(self.device) )
                # the mask codes is to do use the average of results of top-k concepts
                # top_k_values, _ = torch.topk(l_i, 5, dim=1)
                # l_i = top_k_values.mean(dim=1)
                l_i, _ = torch.max(l_i, dim=1)
                logits.append(l_i)
            logits = torch.stack(logits, dim=0)  # batchsize, lan_p_set, num of clsses
        return logits

    def zero_shot_concept_avg(self, image, mask=True):
        with torch.no_grad():
            prompts = self.combined_list.to(self.device) # K, C, Dim
            # using mask
            if mask:
                mask_tensor = self.concept_mask_tensor.unsqueeze(-1).to(self.device)
                prompts = prompts * mask_tensor
                prompts = prompts.sum(dim=1)  #
                count_valid = mask_tensor.sum(dim=1)
                count_valid = count_valid.clamp(min=1)  # avoid to div by 0
                prompts = prompts / count_valid
            else:
                prompts = prompts.mean(dim=1)

            prompts = prompts / prompts.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()

            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 64, 1024

            logits = []
            for img_i in image_features:
                l_i = logit_scale * img_i @ prompts.t()
                # l_i = prompts@img_i# K,C
                # l_i = logit_scale * torch.sum(l_i * self.concept_prior.squeeze(), dim=-1)
                logits.append(l_i)
            logits = torch.stack(logits, dim=0)  # batchsize, lan_p_set, num of clsses
        return logits

    def zero_shot_concept_map_prior(self, image, tau=1.1):
        with torch.no_grad():
            # 1000,100,758
            prompts = self.combined_list.to(self.device)
            logit_scale = self.logit_scale.exp()
            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 64, 1024
            logits = []
            for img_i in image_features:
                l_i = logit_scale * prompts @ img_i  # num of classes, num of concepts
                # 使用最大的簇中心
                # l_i = l_i[torch.arange(l_i.size(0)),  self.map_cid]

                # Select elements based on index_list 选用对应cluster的averaging
                # selected_elements = [torch.mean(l_i[i, torch.tensor(indices)]) for i, indices in enumerate(self.map_cids)]
                # Convert list of tensors to a single tensor (if needed)
                # l_i = torch.stack(selected_elements)

                clip_ability = torch.softmax(l_i / tau, dim=-1)
                l_i = torch.sum(clip_ability * l_i, dim=-1)
                logits.append(l_i)
            logits = torch.stack(logits, dim=0)  # batchsize, lan_p_set, num of clsses
        return logits

    def zero_shot_concept_mad_noise_two_side_huber(self, image, lambda_threshold=2.5, delta=0.05):
        """
        使用双边Huber函数的MAD noise处理方法
        
        Args:
            image: 输入图像
            lambda_threshold: 异常值判定阈值，默认2.5
            delta: 置信度参数，默认0.05
            
        Returns:
            logits: 经过noise-robust处理的预测分数
        """
        with torch.no_grad():
            # 获取所有concept的embeddings并归一化
            prompts = self.combined_list.to(self.device)  # K, C, Dim
            prompts = prompts / prompts.norm(dim=-1, keepdim=True)
            
            # 编码图像特征并归一化
            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # B, Dim
            
            # 计算所有图像与所有concept的相似度
            # logit_scale = self.logit_scale.exp()
            B, K, C, D = image_features.shape[0], prompts.shape[0], prompts.shape[1], prompts.shape[2]
            
            # 重塑图像特征以进行批处理计算
            image_features = image_features.unsqueeze(1).unsqueeze(2)  # B, 1, 1, D
            prompts = prompts.unsqueeze(0)  # 1, K, C, D
            
            # 计算所有相似度 (B, K, C)
            # similarities = logit_scale * torch.matmul(image_features, prompts.transpose(-2, -1)).squeeze(2)
            similarities = torch.matmul(image_features, prompts.transpose(-2, -1)).squeeze(2)
            # 计算每个batch和类别的中位数 (B, K, 1)
            median = similarities.median(dim=-1, keepdim=True)[0]
            
            # 计算MAD (B, K, 1)
            mad = (similarities - median).abs().median(dim=-1, keepdim=True)[0] + 1e-6
            
            # 计算异常值比例 rhohat (B, K, 1)
            outliers = (similarities - median).abs() > lambda_threshold * mad
            rhohat = outliers.float().mean(dim=-1, keepdim=True)  # (B, K, 1)
            rhohat = torch.maximum(rhohat, torch.ones_like(rhohat) / C)  # 确保不小于1/C
            
            # 计算kappa (B, K, 1)
            kappa = torch.sqrt(2 * torch.log(torch.tensor(2.0/delta)) / (C * rhohat))
            kappa = torch.maximum(kappa, torch.ones_like(kappa))  # 确保不小于1
            
            # 计算tau
            tau = kappa * mad
            
            # 计算归一化偏差
            u = (similarities - median) / tau
            
            # 应用Huber's psi函数
            psi_u = self.huber_psi(u)
            
            # 计算最终估计 (B, K)
            logits = median.squeeze(-1) + (tau.squeeze(-1) / C) * psi_u.sum(dim=-1)
            
        return logits
        
    def huber_psi(self, u):
        """
        Huber's psi function: sign(u) * min(|u|, 1)
        
        Args:
            u: 输入张量
        Returns:
            psi(u)
        """
        return torch.sign(u) * torch.minimum(u.abs(), torch.ones_like(u))
        
    def zero_shot_concept_mad_noise(self, image, tau=1.0, lambda_threshold=2.5):
        """
        使用MAD (Median Absolute Deviation) 的noise learning方法
        
        Args:
            image: 输入图像
            tau: 温度参数
            
        Returns:
            logits: 经过noise-robust处理的预测分数
        """
        with torch.no_grad():
            # 获取所有concept的embeddings并归一化
            prompts = self.combined_list.to(self.device)  # K, C, Dim
            prompts = prompts / prompts.norm(dim=-1, keepdim=True)
            
            # 编码图像特征并归一化
            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # B, Dim
            
            # 计算所有图像与所有concept的相似度
            logit_scale = self.logit_scale.exp()
            B, K, C, D = image_features.shape[0], prompts.shape[0], prompts.shape[1], prompts.shape[2]
            
            # 重塑图像特征以进行批处理计算
            image_features = image_features.unsqueeze(1).unsqueeze(2)  # B, 1, 1, D
            prompts = prompts.unsqueeze(0)  # 1, K, C, D
            
            # 计算所有相似度 (B, K, C)
            similarities = logit_scale * torch.matmul(image_features, prompts.transpose(-2, -1)).squeeze(2) / tau
            # similarities = 1/2 *( torch.matmul(image_features, prompts.transpose(-2, -1)).squeeze(2)+1) 
            # 计算每个batch和类别的中位数 (B, K, 1)
            median = similarities.median(dim=-1, keepdim=True)[0]
            
            # 计算MAD (B, K, 1)
            mad = (similarities - median).abs().median(dim=-1, keepdim=True)[0] + 1e-6
            
            # 计算异常值比例 rhohat (B, K)
            # outliers = (similarities - median) < -lambda_threshold * mad
            outliers = (similarities - median).abs() > lambda_threshold * mad
            rhohat = outliers.float().mean(dim=-1, keepdim=True)  # (B, K, 1)
            
            # 计算kappa (B, K, 1)
            kappa = torch.log((1 - rhohat) / rhohat).clamp(0.1, 4.0)
            
            # 计算权重 (B, K, C)
            # z = -kappa * (similarities - median) / mad
            z = -kappa * (similarities - median) / mad
            weights = 1 / (1 + torch.exp(z))
            
            # 计算加权平均 (B, K)
            weighted_sims = weights * similarities
            # Note: similarities already includes logit_scale (line 325), so we don't multiply again
            logits = weighted_sims.sum(dim=-1) / weights.sum(dim=-1)
            
        return logits, rhohat.squeeze(-1)

    def zero_shot_concept_hard_trim(self, image, contamination_rate=0.1, tau=1.0):
        """
        Baseline 1: Hard trim method with fixed contamination rate
        
        Args:
            image: 输入图像
            contamination_rate: 污染率，默认0.1（去掉最低的10%）
            tau: 温度参数
            
        Returns:
            logits: 经过hard trim处理的预测分数
        """
        with torch.no_grad():
            # 获取所有concept的embeddings并归一化
            prompts = self.combined_list.to(self.device)  # K, C, Dim
            prompts = prompts / prompts.norm(dim=-1, keepdim=True)
            
            # 编码图像特征并归一化
            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # B, Dim
            
            # 计算所有图像与所有concept的相似度
            logit_scale = self.logit_scale.exp()
            B, K, C, D = image_features.shape[0], prompts.shape[0], prompts.shape[1], prompts.shape[2]
            
            # 重塑图像特征以进行批处理计算
            image_features = image_features.unsqueeze(1).unsqueeze(2)  # B, 1, 1, D
            prompts = prompts.unsqueeze(0)  # 1, K, C, D
            
            # 计算所有相似度 (B, K, C)
            similarities = logit_scale * torch.matmul(image_features, prompts.transpose(-2, -1)).squeeze(2) / tau
            
            # 对每个类别的concept相似度进行排序（升序）
            sorted_sim, _ = torch.sort(similarities, dim=-1)  # B, K, C
            
            # 计算要修剪的数量（去掉最低的contamination_rate比例）
            num_trim = int(C * contamination_rate)
            
            if num_trim > 0:
                # 去掉最低的num_trim个concept，保留剩余的
                trimmed_sim = sorted_sim[..., num_trim:]
            else:
                trimmed_sim = sorted_sim
            
            # 计算平均值得到最终分数 (B, K)
            logits = trimmed_sim.mean(dim=-1)
            
        return logits

    def zero_shot_concept_median(self, image, tau=1.0):
        """
        Baseline 2: Directly use median as the final score
        
        Args:
            image: 输入图像
            tau: 温度参数
            
        Returns:
            logits: 使用中位数作为预测分数
        """
        with torch.no_grad():
            # 获取所有concept的embeddings并归一化
            prompts = self.combined_list.to(self.device)  # K, C, Dim
            prompts = prompts / prompts.norm(dim=-1, keepdim=True)
            
            # 编码图像特征并归一化
            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # B, Dim
            
            # 计算所有图像与所有concept的相似度
            logit_scale = self.logit_scale.exp()
            B, K, C, D = image_features.shape[0], prompts.shape[0], prompts.shape[1], prompts.shape[2]
            
            # 重塑图像特征以进行批处理计算
            image_features = image_features.unsqueeze(1).unsqueeze(2)  # B, 1, 1, D
            prompts = prompts.unsqueeze(0)  # 1, K, C, D
            
            # 计算所有相似度 (B, K, C)
            similarities = logit_scale * torch.matmul(image_features, prompts.transpose(-2, -1)).squeeze(2) / tau
            
            # 直接使用中位数 (B, K)
            logits = similarities.median(dim=-1)[0]
            
        return logits

    def zero_shot_concept_cauchy(self, image, tau=1.0, scale_factor=1.0):
        """
        Baseline 3: Symmetric Cauchy distribution for robust estimation
        
        Args:
            image: 输入图像
            tau: 温度参数
            scale_factor: Cauchy分布的尺度因子，默认1.0
            
        Returns:
            logits: 使用Cauchy权重处理的预测分数
        """
        with torch.no_grad():
            # 获取所有concept的embeddings并归一化
            prompts = self.combined_list.to(self.device)  # K, C, Dim
            prompts = prompts / prompts.norm(dim=-1, keepdim=True)
            
            # 编码图像特征并归一化
            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # B, Dim
            
            # 计算所有图像与所有concept的相似度
            logit_scale = self.logit_scale.exp()
            B, K, C, D = image_features.shape[0], prompts.shape[0], prompts.shape[1], prompts.shape[2]
            
            # 重塑图像特征以进行批处理计算
            image_features = image_features.unsqueeze(1).unsqueeze(2)  # B, 1, 1, D
            prompts = prompts.unsqueeze(0)  # 1, K, C, D
            
            # 计算所有相似度 (B, K, C)
            similarities = logit_scale * torch.matmul(image_features, prompts.transpose(-2, -1)).squeeze(2) / tau
            
            # 计算中位数和MAD作为位置和尺度参数
            median = similarities.median(dim=-1, keepdim=True)[0]  # (B, K, 1)
            mad = (similarities - median).abs().median(dim=-1, keepdim=True)[0] + 1e-6  # (B, K, 1)
            
            # 计算Cauchy权重: w = 1 / (1 + ((x - median) / scale)^2)
            # 使用MAD作为尺度参数
            scale = scale_factor * mad
            normalized_diff = (similarities - median) / scale  # (B, K, C)
            weights = 1 / (1 + normalized_diff ** 2)  # (B, K, C)
            
            # 计算加权平均 (B, K)
            weighted_sims = weights * similarities
            logits = weighted_sims.sum(dim=-1) / weights.sum(dim=-1)
            
        return logits


        # u = (s - m) / tau


    def zero_shot_concept_avg_noise(self, image, alpha=0.2):
        """
        使用noise label learning的方式增强对noise concept的适应，使用矩阵操作进行批处理
        
        Args:
            image: 输入图像
            alpha: 裁剪比例，会移除前后alpha比例的concept
            
        Returns:
            logits: 经过noise-robust处理的预测分数
        """
        with torch.no_grad():
            # 获取所有concept的embeddings并归一化
            prompts = self.combined_list.to(self.device)  # K, C, Dim
            prompts = prompts / prompts.norm(dim=-1, keepdim=True)
            
            # 编码图像特征并归一化
            image_features = self.clip.encode_image(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # B, Dim
            
            # 计算所有图像与所有concept的相似度
            logit_scale = self.logit_scale.exp()
            B, K, C, D = image_features.shape[0], prompts.shape[0], prompts.shape[1], prompts.shape[2]
            
            # 重塑图像特征以进行批处理计算
            image_features = image_features.unsqueeze(1).unsqueeze(2)  # B, 1, 1, D
            prompts = prompts.unsqueeze(0)  # 1, K, C, D
            
            # 计算所有相似度 (B, K, C)
            similarities = logit_scale * torch.matmul(image_features, prompts.transpose(-2, -1)).squeeze(2)
            
            # 对每个类别的concept相似度进行排序
            sorted_sim, _ = torch.sort(similarities, dim=-1)  # B, K, C
            
            # 计算每个类别要裁剪的数量
            num_concepts = sorted_sim.shape[-1]
            num_trim = int(num_concepts * alpha)
            
            if num_trim > 0:
                # 使用切片操作一次性裁剪所有batch和类别的相似度
                trimmed_sim = sorted_sim[..., num_trim:-num_trim]
            else:
                trimmed_sim = sorted_sim
            
            # 计算平均值得到最终分数 (B, K)
            logits = trimmed_sim.mean(dim=-1)
            
        return logits


    def forward(self, image):
        prompts = self.combined_list.to(self.device)
        prompts = prompts / prompts.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        image_features = self.clip.encode_image(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 64, 1024

        logits = []
        for img_i in image_features:  # here img_i uis
            l_i = logit_scale * img_i @ prompts.t()  #
            logits.append(l_i)
        logits = torch.stack(logits, dim=0)  # batchsize, num of clsses
        return logits

    def forward_concept(self, image):
        prompts = self.combined_list.to(self.device)  # K, C, Dim
        # using mask
        # if mask:
        #     mask_tensor = self.concept_mask_tensor.unsqueeze(-1).to(self.device)
        #     prompts = prompts * mask_tensor
        #     prompts = prompts.sum(dim=1)  #
        #     count_valid = mask_tensor.sum(dim=1)
        #     count_valid = count_valid.clamp(min=1)  # avoid to div by 0
        #     prompts = prompts / count_valid
        # else:
        #     prompts = prompts.mean(dim=1)
        prompts = prompts / prompts.norm(dim=-1, keepdim=True)

        image_features = self.clip.encode_image(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # N, Dim

        # Matrix multiplication: (K, C, Dim) @ (N, Dim) -> (N, K, C)
        # Using einsum for efficient batch computation
        logits = torch.einsum('kcd,nd->nkc', prompts, image_features)  # N, K, C. N is the number of views
        # print(logits.shape)
        return logits
    def forward_concept_time_stats(self, prompts, image):

        image_features = self.clip.encode_image(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # N, Dim

        # Matrix multiplication: (K, C, Dim) @ (N, Dim) -> (N, K, C)
        # Using einsum for efficient batch computation
        logits = torch.einsum('kcd,nd->nkc', prompts, image_features)  # N, K, C. N is the number of views
        # print(logits.shape)
        return logits

    def forward_concept_batch(self, image):
        image_features = self.clip.encode_image(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 64, 1024

        # logits = (self.prompts @ image_features.t()).permute(2, 0, 1)
        # logits = torch.einsum('bij,kj->ikb', self.prompts, image_features)
        prompts_reshaped = self.prompts.view(-1, 512)  # [10000, 512]

        # 矩阵乘法
        logits = prompts_reshaped @ image_features.t()  # [10000, 64]

        # 还原维度并调整顺序
        logits = logits.view(100, 100, 64).permute(2, 0, 1)  # [64, 100, 100]
        return logits

    def forward_concept_feature(self, image):
        prompts = self.combined_list.to(self.device)  # K, C, Dim
        prompts = prompts / prompts.norm(dim=-1, keepdim=True) # K, C, Dim

        image_features = self.clip.encode_image(image.type(self.dtype))  # N, dim
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 64, 1024
        return image_features, prompts, self.logit_scale.exp()
