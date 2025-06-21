"""
Models package for packet-based malware detection.
"""

from .vit_packet import (
    VisionTransformer,
    vit_tiny_patch16_64,
    vit_small_patch16_128,
    vit_base_patch16_224,
    vit_packet_small,
    vit_packet_base,
    create_model,
    list_models,
    MODEL_REGISTRY
)

from .patch_embed import (
    PatchEmbedding,
    HybridPatchEmbedding,
    AdaptivePatchEmbedding,
    ConvStemPatchEmbedding,
    create_patch_embedding
)

from .position_embed import (
    PositionEmbedding,
    ByteOrderAwarePositionEmbedding,
    RelativePositionEmbedding,
    InterpolatedPositionEmbedding,
    create_position_embedding
)

__all__ = [
    # Main models
    'VisionTransformer',
    'vit_tiny_patch16_64',
    'vit_small_patch16_128',
    'vit_base_patch16_224',
    'vit_packet_small',
    'vit_packet_base',
    'create_model',
    'list_models',
    'MODEL_REGISTRY',
    
    # Patch embeddings
    'PatchEmbedding',
    'HybridPatchEmbedding',
    'AdaptivePatchEmbedding',
    'ConvStemPatchEmbedding',
    'create_patch_embedding',
    
    # Position embeddings
    'PositionEmbedding',
    'ByteOrderAwarePositionEmbedding',
    'RelativePositionEmbedding',
    'InterpolatedPositionEmbedding',
    'create_position_embedding',
]