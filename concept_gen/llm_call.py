import time

from openai import OpenAI
import logging
from pydantic import BaseModel
import json
from data.imagnet_prompts import imagenet_classes

import re

logger = logging.getLogger()


def read_api_keys(file_path=None):
    """Load API keys from concept_gen/api_key.txt. Uses api_config for consistent path."""
    from concept_gen.api_config import load_api_keys
    return load_api_keys()
# use gpt 4o  gpt 4.0 toru bo"gpt-4o"


# json schema for gpt-4o
class Class(BaseModel):
    classname: str
    concepts: list[str]

# class ConceptForEachClass(BaseModel):
#     classes: list[Class]
#
# json_schema_classes = { "type": "json_schema",
#                     "json_schema":{
#   "type": "object",
#   "strict": True,
#   "properties": {
#     "classes": {
#       "type": "array",
#       "items": {
#         "type": "object",
#         "properties": {
#           "classname": {
#             "type": "string",
#             "description": "The classname of images",
#           },
#           "concepts": {
#             "type": "array",
#             "items": {
#               "type": "string"
#             },
#             "description": "The most possible concepts corresponding to classname",
#           }
#         },
#         "required": ["classname", "concepts"]
#       }
#     }
#   },
#   "required": ["classes"]
# }
# }

json_schema_classes = { "type": "json_schema",
                    "json_schema":{
"name": "classes_concepts",
"schema": {
  "type": "object",
  "strict": True,
  "properties": {
    "classes": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "classname": {
            "type": "string",
            "description": "The classname of images",
          },
          "concepts": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "The most possible concepts corresponding to classname",
          }
        },
        "required": ["classname", "concepts"],
"additionalProperties": False
      }
    }
  },
  "required": ["classes"],
"additionalProperties": False
},
                        "strict": True
                    }
}


class GPTAPI:
    def __init__(self, engine):
        # "gpt-4o"  "gpt-4-turbo" "gpt-4o-mini"
        self.engine = engine
        keys = read_api_keys()
        if not keys:
            raise ValueError("api_key.txt is empty. Add your API key on line 1.")
        self.api_key = keys[0]
        self.client = OpenAI(api_key=self.api_key)

    def call_api(self, messages, n, temperature=1):
        """"""
        result = None
        while result is None:
            try:
                # Structured Outputs are available for gpt-4o, other  models like gpt-4-turbo and earlier may use
                # JSON mode instead. We do not use Structured Outputs nor json mode for compatibility with other models.
                result = self.client.chat.completions.create(
                    model=self.engine,
                    messages=messages,
                    n=n,
                    temperature=temperature
                )
                result = [r.message.content for r in result.choices]
            except Exception as e:
                # to solve rate limit error
                logger.info(f"{str(e)}", "Retry.")
                time.sleep(5)
        return result

    def parse_result(self, result):
        valid_concept = []
        for r in result:
            last_line = r.strip().split('\n')[-1].strip().lower()
            # Check if "a photo of" is in the last line
            if "a photo of" in last_line:
                match = re.search(r"with (.+)", last_line, re.IGNORECASE)
                # if match:
                #     concept = match.group(1).strip()
                #     concept = re.sub(r"[.,;]+$", "", concept)
                #     valid_concept.append(concept )
            else:
                match = re.search(r"the final concept is: (.+)", last_line, re.IGNORECASE)
            if match:
                concept = match.group(1).strip()
                concept = re.sub(r"[.,;*\"]+$", "", concept).strip()
                valid_concept.append(concept)
        return valid_concept



if __name__ == '__main__':
    pass

#     system_prompt = """You are a concept proposer specializing in enhancing image classification tasks using CLIP. Your goal is to generate class-specific concepts that improve the distinguishability of classes. Each class is represented by a text description (e.g., “A photo of {class}.”), and your task is to enrich these descriptions with effective concepts.
#
# Guidelines:
#
# Understand the Class Context: Analyze the unique features and characteristics of each class (e.g., “cat”) to identify potential distinguishing concepts.
# Concept Format: Propose concepts that can be integrated into the fixed template: “A photo of {class} with {concept}.” Ensure these concepts highlight features that make the class distinct.
# Maximize Similarity: The chosen concepts should enhance the text encoding's similarity to the image representation, improving classification accuracy.
# Diversity: Consider a wide range of concepts, focusing on relevance and the ability to differentiate the class from others while providing diverse concepts for one class.
# Understand the ability of CLIP: CLIP is pre-trained on image-text pairs from the Internet, whose text includes high-frequency words from Wikipedia and WordNet synsets, covering a broad set of visual concepts.
#
# Example Implementation:
# For the class “cat,” consider concepts like “fluffy fur” to create the text “An photo of a cat with fluffy fur.”
# """
#     user_prompt = """The class set includes [{}]. Please generate top-{} concept for each class using the given json scema mode."""
#     user_prompt = user_prompt.format(", ".join(imagenet_classes[0:5]), 1)
#     messages = [
#         {"role": "system",
#          "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]
#
# gpt_engine = GPT_API()
# result = gpt_engine.call_api_structured(n=10, messages=messages)

# """You are a concept proposer specializing in enhancing image classification tasks using CLIP. Your goal is to generate class-specific concepts that improve the distinguishability of classes. Each class is represented by a text description (e.g., “A photo of {class}.”), and your task is to enrich these descriptions with effective concepts.
#
# Guidelines:
#
# Understand the Class Context: Analyze the unique features and characteristics of each class (e.g., “cat”) to identify potential distinguishing concepts.
# Concept Integration: Propose concepts that can be integrated into the fixed template: “A photo of {class} with {concept}.” Ensure these concepts highlight features that make the class distinct.
# Maximize Similarity: The chosen concepts should enhance the text encoding's similarity to the image representation, improving classification accuracy.
# Diversity: Consider a wide range of concepts, focusing on relevance and the ability to differentiate the class from others while giving diverse concepts for one class.
# Understand the ability of CLIP:
# Example Implementation:
# For the class “cat,” consider concepts like “with fluffy fur” or “with green eyes” to create the text “An image of a cat with fluffy fur.”
# Iterative Refinement: Review and refine concepts based on feedback and classification outcomes to ensure continuous improvement"""

#
# """Large pre-trained vision-language models such as CLIP can be utilized for image classification. Given a fixed set of classes (e.g., “cat”),  to classify an image, each class is mapped to  a fixed template (e.g., “An image of {class}”) and encoded by the CLIP text encoder. And the image is encoded by the CLIP image encoder.
# The predication for this image is the class whose text description has the highest similarity to the image representation. This classification process can be enhanced by added class-specific concepts into the fixed template as (e.g., “An image of {class} with {concept}”). These concepts need to cater to the ability of and make corresponding class distinguishable with other classes if available.
#
# Please image you are a concept proposer to give the most effective concept. """



# system_prompt = """You are a concept proposer specializing in enhancing image classification tasks using CLIP. Your goal is to generate class-specific concepts that improve the distinguishability of classes. Each class is represented by a text description (e.g., “A photo of {class}.”), and your task is to enrich these descriptions with effective concepts.
#
# Guidelines:
# Understand the ability of CLIP: CLIP is pre-trained on image-text pairs from the Internet, whose text includes high-frequency words from Wikipedia and WordNet synsets, covering a broad set of visual concepts. It may cannot process two much sophisticated and nuanced concept.s
#
# Understand the Class Context: Analyze the unique features and characteristics of each class (e.g., “cat”) to identify potential distinguishing concepts.
# Concept Format: Propose concepts that can be integrated into the fixed template: “A photo of {class} with {concept}.” Ensure these concepts highlight features that make the class distinct.
# Maximize Similarity: The chosen concepts should enhance the text encoding's similarity to the image representation, improving classification accuracy.
# Diversity: Consider a wide range of concepts, focusing on relevance and the ability to differentiate the class from others while providing diverse concepts for one class.
# Understand the ability of CLIP: CLIP is pre-trained on image-text pairs from the Internet, whose text includes high-frequency words from Wikipedia and WordNet synsets, covering a broad set of visual concepts.
#
# Example Implementation:
# For the class “cat,” consider concepts like “fluffy fur” to create the text “An photo of a cat with fluffy fur.”
#
#
# Guidelines:
#
# Understand the Class Context:
# Analyze the unique features and characteristics of each class to identify distinguishing concepts.
# Maximize Similarity:
# Choose concepts that improve the similarity between text and image embeddings.
# Leverage CLIP's Capabilities:
# CLIP is trained on diverse image-text pairs with high-frequency words from sources like Wikipedia and WordNet. Avoid overly nuanced concepts.
# Concept Format:
# Use the format: “A photo of {class} with {concept}.”
# Highlight features that make the class distinct.
# """