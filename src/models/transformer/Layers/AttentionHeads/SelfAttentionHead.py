import tensorflow as tf
from src.models.Transformer.Layers.AttentionHeads.BaseAttention import BaseAttention


class SelfAttentionHead(BaseAttention):
    """
        Layer that implements self attention head
        Assignemtn of KQV is all the input x
    """
    def call(self, x):
        attention_output = self.multi_head_attn(query = x, value = x, key = x,use_causal_mask = True)
        x = self.add([x, attention_output])
        x = self.layer_norm(x)
        return x