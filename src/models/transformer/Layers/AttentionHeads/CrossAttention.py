import tensorflow as tf
from src.models.Transformer.Layers.AttentionHeads.BaseAttention import BaseAttention


class CrossAttention(BaseAttention):
    def call(self, x, context):
        #Assigning 2 here is very important!!!!!
        attention_output, attn_scores = self.multi_head_attn(query=x,key=context,value=context,return_attention_scores=True) 
        # Cache the attention scores for plotting later.
        self.last_attn_scores = attention_output

        x = self.add([x, attention_output])
        x = self.layer_norm(x)
        return x