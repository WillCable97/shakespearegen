import tensorflow as tf
from src.models.Transformer.Layers.AttentionHeads.BaseAttention import BaseAttention

class GlobalSelfAttention(BaseAttention):
    """
        Self attention without Masking
    """
    def call(self, x, mask=None):
        attention_output = self.multi_head_attn(query=x,value=x,key=x, attention_mask = mask)
        x = self.add([x, attention_output])
        x = self.layer_norm(x)
        return x