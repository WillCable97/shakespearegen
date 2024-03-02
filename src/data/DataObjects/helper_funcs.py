

#Maps for data procesing
def create_offset_labels(input_tensor):
    return input_tensor[:-1], input_tensor[1:]


def reorder_transformer_dataset(context_tensor, content_tensor):
   a = context_tensor
   (b, c) = content_tensor
   return (a,b) , c

