"""
Model registry for skill classification architectures.

To add a new model:
    1. Create a new file in skill_segmentor/models/
    2. Decorate the model class with @register_model("name")
    3. Import it in this file
"""

MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


# Import all model files so they register themselves
from skill_classifier.models.mlp import *  # noqa
from skill_classifier.models.transformer import *  # noqa
from skill_classifier.models.tcn import *  # noqa
from skill_classifier.models.tcn_supcon import *  # noqa
from skill_classifier.models.spatial_attention_mlp import *  # noqa
