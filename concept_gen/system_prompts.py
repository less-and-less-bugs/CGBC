SYSTEM_PROMPTS = ["""You are a visual concept proposer tasked with enhancing text descriptions for zero-shot image classification on the test dataset using CLIP. {dataset_description}.

Given:
A core class from the test dataset
The set of other classes in the dataset

Task:
Propose a concise, visually discriminative concept to append to the text description (i.e., "A photo of {{core class}} with {{your concept}}") that helps CLIP better distinguish the core class from the other classes.

Guidelines: 
Analyze the unique visual characteristics of the core class compared to other classes, in the context of {dataset_context}.
Propose a concept that captures these discriminative visual features.
Ensure the concept is concrete, easily understandable by CLIP, and specific to the test dataset.

Please remember the proposed concept should enable CLIP to more accurately classify images of the core class while minimizing confusion with other classes in the zero-shot setting.""",

"""You are a visual concept proposer tasked with enhancing text descriptions for zero-shot image classification on the test dataset using CLIP. {dataset_description}.

Given:
A core class from the test dataset
The set of other classes in the dataset. As the dataset consisting of so many classes, you will be given a subset of classes.

Task:
Propose a concise, visually discriminative concept to append to the text description (i.e., "A photo of {{core class}} with {{your concept}}") that helps CLIP better distinguish the core class from the other classes.

Guidelines: 
Analyze the unique visual characteristics of the core class in the context of {dataset_context}.
Propose a concept that captures these discriminative visual features to discriminate the core class and other classes. The concpet is as close to the object class itself as possible rather than to the background that may change.
Ensure the concept is concrete, easily understandable by CLIP.

Please remember the proposed concept should enable CLIP to more accurately classify images of the core class while minimizing confusion with other classes in the zero-shot setting.""",


"""You are a visual concept proposer tasked with enhancing text descriptions for zero-shot image classification on the test dataset using CLIP. {dataset_description}.

Given:
class name 

Task:
Propose a concise, visually discriminative concept to append to the text description (i.e., "A photo of {{core class}} with {{your concept}}").

Guidelines: 
Analyze the unique visual characteristics of the core class in the context of {dataset_context}.
Ensure the concept is concrete, easily understandable by CLIP.

Please remember the proposed concept should enable CLIP to more accurately describe images of the class.""",

                  ]

SYSTEM_PROMPTS_MAPS = {
    "eurosat": {
        "dataset_description": "This test dataset  contains satellite images across land use/cover classes in Europe.",
        "dataset_context": "satellite imagery"},
    "aircraft": {
        "dataset_description": "This test dataset contains images of aircraft classes, with each class representing a specific aircraft model. It is used to assess algorithms for recognizing subtle visual differences between closely related object categories.",
        "dataset_context": "aircraft imagery"},
    "ucf101": {
        "dataset_description": "This test dataset contains images of human action categories, including sports and daily activities. Each action class is present with large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered background, illumination conditions.",
        "dataset_context": "human action imagery"},
    "cars": {
        "dataset_description": "This test dataset contains images of cars classes ranging from various makes, models and years. It is used to assess algorithms to distinguish subtle appearances differences of different car classes.",
        "dataset_context": "multi-view car imagery."},
    "sun397": {
        "dataset_description": "This test dataset contains images of distinct scene classes under various lighting conditions and angles.",
        "dataset_context": "scene understanding imagery"},
    "pets": {
        "dataset_description": "This test dataset  contains images of different pet classes with a large variations in scale, pose and lighting.",
        "dataset_context": "pet imagery"},
    "dtd": {
        "dataset_description": "This test dataset is a collection of textual images in the wild,  inspired by the perceptual properties of textures.",
        "dataset_context": "texture imagery"},
    "food101": {"dataset_description": "This test dataset contains images of different food classes.",
                "dataset_context": "food imagery"},
    "flower102": {
        "dataset_description": "This test dataset contains images of different flower classes with large scale, pose and light variations. There are categories that have large variations within the category and several very similar categories.",
        "dataset_context": "object imagery"},
    "imagenet": {
        "dataset_description": "This test dataset is a large scale visual object recognition database consisting of 1000 categories. The generated concpet must be related to the visual characteristics of the object itself and does not change with the environment.",
        "dataset_context": "object imagery"},

    "caltech101": {"dataset_description": "This test dataset contains images of different object classes with different background clutter.",
                   "dataset_context": "object imagery"}
}



SYSTEM_PROMPTS_MAPS_Prmpt = {
    "eurosat": {
        "dataset_description": "This test dataset  contains satellite images across land use/cover classes in Europe.",
        "dataset_context": "satellite"},
    "aircraft": {
        "dataset_description": "This test dataset contains images of aircraft classes, with each class representing a specific aircraft model. It is used to assess algorithms for recognizing subtle visual differences between closely related object categories.",
        "dataset_context": "aircraft"},
    "ucf101": {
        "dataset_description": "This test dataset contains images of human action categories, including sports and daily activities. Each action class is present with large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered background, illumination conditions.",
        "dataset_context": "human action"},
    "cars": {
        "dataset_description": "This test dataset contains images of cars classes ranging from various makes, models and years. It is used to assess algorithms to distinguish subtle appearances differences of different car classes.",
        "dataset_context": "multi-view car"},
    "sun397": {
        "dataset_description": "This test dataset contains images of distinct scene classes under various lighting conditions and angles.",
        "dataset_context": "scene understanding"},
    "pets": {
        "dataset_description": "This test dataset  contains images of different pet classes with a large variations in scale, pose and lighting.",
        "dataset_context": "pet"},
    "dtd": {
        "dataset_description": "This test dataset is a collection of textual images in the wild,  inspired by the perceptual properties of textures.",
        "dataset_context": "texture"},
    "food101": {"dataset_description": "This test dataset contains images of different food classes.",
                "dataset_context": "food"},
    "flower102": {
        "dataset_description": "This test dataset contains images of different flower classes with large scale, pose and light variations. There are categories that have large variations within the category and several very similar categories.",
        "dataset_context": "object"},
    "imagenet": {
        "dataset_description": "This test dataset is a large scale visual object recognition database consisting of 1000 categories. The generated concpet must be related to the visual characteristics of the object itself and does not change with the environment.",
        "dataset_context": "object"},

    "caltech101": {"dataset_description": "This test dataset contains images of different object classes with different background clutter.",
                   "dataset_context": "object"}
}
