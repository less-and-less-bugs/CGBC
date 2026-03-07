"""
This file implements batch concept sampling using LLM and CLIP text encoder for filtering.
"""
import json
import asyncio
from typing import List, Dict, Any, Optional
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from sentence_transformers import SentenceTransformer
import re
import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concept_gen.llm_provider import LLMProvider, ProviderType, GenerationConfig
from concept_gen.api_config import load_api_config
from data.cls_to_names import *
from data.imagnet_prompts import imagenet_classes
import logging
from concept_gen.system_prompts import *
import os

class ConceptBatchSampler:
    def __init__(
        self,
        provider: LLMProvider,
        task_name: str,
        class_names: List[str],
        similarity_model: str = "clip",  # 'clip' or 'sbert'
        clip_model_name: str = "openai/clip-vit-base-patch32",
        sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the concept batch sampler
        
        Args:
            provider: LLM provider instance
            task_name: Name of the dataset/task (e.g., 'eurosat', 'imagenet')
            class_names: List of all class names in the dataset
            similarity_model: Which model to use for similarity computation ('clip' or 'sbert')
            clip_model_name: CLIP model name for text encoding
            sbert_model_name: Sentence-BERT model name for semantic similarity
            device: Device to run models on
        """
        self.provider = provider
        self.task_name = task_name
        self.class_names = class_names
        self.device = device
        self.similarity_model = similarity_model.lower()
        
        if self.similarity_model not in ['clip', 'sbert']:
            raise ValueError("similarity_model must be either 'clip' or 'sbert'")
        
        # Initialize models based on choice
        if self.similarity_model == 'clip':
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(device)
            self.sbert = None
        else:  # sbert
            self.sbert = SentenceTransformer(sbert_model_name, device=device)
            self.tokenizer = None
            self.text_encoder = None
        
        # Load dataset descriptions from system_prompts
        from system_prompts import SYSTEM_PROMPTS_MAPS
        self.dataset_info = SYSTEM_PROMPTS_MAPS.get(task_name, {
            "dataset_description": "This test dataset contains various images for classification",
            "dataset_context": "general imagery"
        })
        
        # Pre-compute class embeddings
        self.class_embeddings = self._compute_class_embeddings()
        
    def _parse_concepts_response(self, text: str) -> Optional[List[str]]:
        """Parse LLM response to extract concepts"""
        try:
            # Extract content between markers
            start_marker = "<concepts begin>"
            end_marker = "</concepts end>"
            
            start_idx = text.index(start_marker)
            end_idx = text.index(end_marker)
            content = text[start_idx + len(start_marker):end_idx].strip()
            
            # Extract concepts
            pattern = r"The final concept is:\s*(.*?)(?=(?:\n|$))"
            matches = re.findall(pattern, content)
            
            concepts = []
            for concept in matches:
                concept = concept.strip()
                # Remove brackets if present
                concept = re.sub(r'^\[|\]$', '', concept).strip()
                if concept:
                    concepts.append(concept)
            return concepts
            
        except Exception as e:
            print(f"Error parsing concepts response: {str(e)}")
            return None
            
    def _compute_class_embeddings(self) -> Dict[str, torch.Tensor]:
        """Pre-compute embeddings for all classes"""
        embeddings = {}
        for class_name in self.class_names:
            if self.similarity_model == 'sbert':
                # Get SBERT embedding
                embedding = self.sbert.encode(class_name, convert_to_tensor=True)
                if len(embedding.shape) == 1:
                    embedding = embedding.unsqueeze(0)
            else:  # clip
                # Get CLIP embedding
                inputs = self.tokenizer(f"A photo of {class_name}", return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.text_encoder(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    # Normalize embedding
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embeddings[class_name] = embedding
        return embeddings
        
    def compute_class_similarity(self, class1: str, class2: str) -> float:
        """Compute semantic similarity between two classes"""
        # Get pre-computed embeddings
        emb1 = self.class_embeddings[class1]
        emb2 = self.class_embeddings[class2]
        
        # Ensure embeddings are 2D
        if len(emb1.shape) == 1:
            emb1 = emb1.unsqueeze(0)
        if len(emb2.shape) == 1:
            emb2 = emb2.unsqueeze(0)
            
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        
        # Return the first (and only) similarity value
        return similarity[0].item()
        
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        if self.similarity_model == 'sbert':
            # Use Sentence-BERT
            emb1 = self.sbert.encode(text1, convert_to_tensor=True)
            emb2 = self.sbert.encode(text2, convert_to_tensor=True)
            
            # Ensure embeddings are 2D
            if len(emb1.shape) == 1:
                emb1 = emb1.unsqueeze(0)
            if len(emb2.shape) == 1:
                emb2 = emb2.unsqueeze(0)
        else:  # clip
            # Use CLIP
            inputs1 = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to(self.device)
            inputs2 = self.tokenizer(text2, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                # Get embeddings and mean pool over sequence length
                outputs1 = self.text_encoder(**inputs1)
                outputs2 = self.text_encoder(**inputs2)
                emb1 = outputs1.last_hidden_state.mean(dim=1)
                emb2 = outputs2.last_hidden_state.mean(dim=1)
                
                # Normalize embeddings
                emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
            
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        return similarity.item()
        
    async def sample_concepts(
        self,
        core_class: str,
        num_concepts: int = 10,
        batch_size: int = 5,
        sampling_window: int = 4,
        similarity_threshold: float = 0.8,
        max_retries: int = 5
    ) -> List[str]:
        """
        Sample concepts using batch generation
        
        Args:
            core_class: The main class to generate concepts for
            num_concepts: Total number of concepts to generate
            batch_size: Number of concepts to generate per LLM call
            sampling_window: Number of other classes to consider for each generation
            similarity_threshold: Threshold for filtering similar concepts
            max_retries: Maximum number of retries for failed generations
            
        Returns:
            List of generated concepts
        """
        system_prompt = f"""You are a visual concept proposer tasked with enhancing text descriptions for zero-shot image classification on the test dataset using CLIP. {self.dataset_info['dataset_description']}.

        Given:
        - A core class from the test dataset
        - The set of other classes in the dataset
        
        Task:
        Propose concise, visually discriminative concepts to append to the text description (i.e., "A photo of {{core class}} with {{your concept}}") that helps CLIP better distinguish the core class from the other classes.
        
        Guidelines:
        - Analyze the unique visual characteristics of the core class compared to other classes, in the context of {self.dataset_info['dataset_context']}.
        - Propose concepts that capture these discriminative visual features.
        - Ensure concepts are concrete, easily understandable by CLIP, and specific to the test dataset.
        - Each concept should enable CLIP to more accurately classify images of the core class while minimizing confusion with other classes.
        
        IMPORTANT: Your response must follow this exact format:
        
        <concepts begin>
        concept1
        concept2
        concept3
        </concepts end>
        
        Rules:
        - Start with <concepts begin> and end with </concepts end>
        - Each concept should be on a new line
        - Each concept MUST start with "The final concept is: "
        - Ensure concepts are clear, specific, and relevant to the core class
        - Avoid generic or ambiguous concepts
        - Each concept should be unique and distinct from others
        -Keep each concept brief (ideally ≤6 words), specific, and easy for CLIP to parse."""
        
        gen_config = {
            "temperature": 0.7,
            # "top_p": 0.95
        }
        
        all_concepts = []
        total_attempts = 0
        
        while len(all_concepts) < num_concepts and total_attempts < max_retries * (num_concepts // batch_size + 1):
            total_attempts += 1
            
            try:
                # Get all other classes and their similarities to core class
                similarities = []
                for cls in self.class_names:
                    if cls != core_class:
                        try:
                            sim = self.compute_class_similarity(core_class, cls)
                            similarities.append((cls, sim))
                        except Exception as e:
                            print(f"Error computing similarity between {core_class} and {cls}: {str(e)}")
                            continue
                
                other_classes = similarities
            
                # Sort by similarity (highest to lowest) and select top classes
                other_classes.sort(key=lambda x: x[1], reverse=True)
                sampled_classes = [cls for cls, _ in other_classes[:sampling_window]]
            
                print(f"Selected similar classes for {core_class}:")
                for cls, sim in other_classes[:sampling_window]:
                    print(f"  - {cls} (similarity: {sim:.3f})")
                
                # Create prompt with sampled classes
                prompt = f"Core class: {core_class}. Other classes: {', '.join(sampled_classes)}. Please generate {batch_size} unique and discriminative concepts."
                
                # Generate batch of concepts
                result = await self.provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    config=gen_config
                )
                
                # Parse concepts
                new_concepts = self._parse_concepts_response(result.text)
                if not new_concepts:
                    continue
                    
                # Filter similar concepts
                for concept in new_concepts:
                    is_similar = False
                    for existing_concept in all_concepts:
                        similarity = self.compute_text_similarity(concept, existing_concept)
                        if similarity > similarity_threshold:
                            is_similar = True
                            print(f"Skipping similar concept (similarity: {similarity:.2f})")
                            break
                            
                    if not is_similar and len(all_concepts) < num_concepts:
                        all_concepts.append(concept)
                        print(f"Added new concept: {concept}")
                    else:
                        print(f"Skipping similar concept {concept} (similarity: {similarity:.2f})")
                    
            except Exception as e:
                print(f"Error in concept generation: {str(e)}")
                continue
                
            if len(all_concepts) >= num_concepts:
                break
                
        print(f"Generated {len(all_concepts)} unique concepts")
        return all_concepts

async def process_dataset(dataset_name: str, api_key_id: int = 0, target_concepts: int = 50, max_attempts: int = 5, model_name: str = "gpt-4.1-2025-04-14", dir="50_sim", sampling_window: int = 10):
    """Process a single dataset"""
    # Setup logging
    log_path = dir
    log_dir = os.path.join("batchconcepts", dataset_name, log_path)
    log_file_path = os.path.join(log_dir, "log.txt")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Load API config from api_key.txt (api_key on line 1, base_url on line 2 optional)
    config = load_api_config(api_key_id=api_key_id)
    
    provider = LLMProvider(
        provider=ProviderType.OPENAI,
        model=model_name,
        config=config
    )
    
    # Import class names dynamically
    try:
        class_names = eval(f"{dataset_name}_classes")
    except ImportError:
        logger.error(f"Could not import class names for dataset {dataset_name}")
        return
    except NameError:
        logger.error(f"Class names not found for dataset {dataset_name}")
        return
    
    logger.info(f"Processing dataset: {dataset_name} with {len(class_names)} classes")
    
    # Initialize sampler
    sampler = ConceptBatchSampler(
        provider=provider,
        task_name=dataset_name,
        class_names=class_names,
        similarity_model="clip"  # 使用CLIP进行相似度计算
    )
    
    # Load existing results if available
    results_file = os.path.join(log_dir, "results.json")
    results = {}
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
            logger.info(f"Loaded existing results with {len(results)} classes")
    
    # Process each class
    for core_class in class_names:
        existing_concepts = results.get(core_class, [])
        remaining_concepts = target_concepts - len(existing_concepts)
        
        if remaining_concepts <= 0:
            logger.info(f"Class {core_class} already has {len(existing_concepts)} concepts, skipping...")
            continue
            
        logger.info(f"Class {core_class} has {len(existing_concepts)} concepts, generating {remaining_concepts} more...")
        
        attempts = 0
        while remaining_concepts > 0 and attempts < max_attempts:
            attempts += 1
            logger.info(f"Attempt {attempts}/{max_attempts} for class {core_class}")
            
            try:
                new_concepts = await sampler.sample_concepts(
                    core_class=core_class,
                    num_concepts=remaining_concepts,
                    batch_size=10,          # Generate 8 concepts per batch
                    sampling_window=sampling_window,     # Consider 10 other classes for each generation
                    similarity_threshold=0.95  # Higher threshold for more diverse concepts
                )
                
                # Add new concepts to existing ones
                if core_class not in results:
                    results[core_class] = []
                results[core_class].extend(new_concepts)
                
                # Update remaining count
                remaining_concepts = target_concepts - len(results[core_class])
                logger.info(f"Added {len(new_concepts)} concepts, {remaining_concepts} remaining")
                
                # Save results after each successful generation
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error generating concepts for {core_class}: {str(e)}")
                continue
        
        if remaining_concepts > 0:
            logger.warning(f"Could not generate all concepts for {core_class} after {max_attempts} attempts. Missing {remaining_concepts} concepts.")
                

    
    logger.info(f"Completed processing dataset: {dataset_name}")
    return results

async def main():
    # List of datasets to process
    datasets = ["eurosat",  "aircraft", "ucf101", "cars", "sun397", "pets", "dtd"] # Add more datasets as needed: "food101", "flower102", "caltech101", "dtd", "aircraft", "ucf101", "cars", "sun397",   "eurosat", "pets", "imagenet" 
    model_name = "gpt-4.1"
    dir = "50_mini"
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        await process_dataset(
            dataset_name=dataset_name,
            model_name=model_name,
            api_key_id=0,
            target_concepts=50,  # 目标每个类50个concepts
            max_attempts=10,       # 最多尝试5次
            dir=dir
        )
if __name__ == "__main__":
    asyncio.run(main())
