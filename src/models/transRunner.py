import numpy as np
import tensorflow as tf






num_layers = 3
d_model = 256
dff = 256
num_heads = 2
dropout_rate = 0.1


transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    target_vocab_size=vocab_s,
    dropout_rate=dropout_rate)





class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)




def pad_list(input_list, max_size):
    list_size = len(input_list)
    if list_size >= max_size: return input_list[:80]
    padding_count = max_size - list_size
    padding_list = [0] * padding_count 
    return input_list + padding_list

def masked_loss(label, pred):
  print(label)
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)


transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])



transformer.fit(batched_dataset,
                epochs=1)


transformer.save_weights("./transformer_v3/transformer_v3")


starting_string = "Rome:"

used_tokenizer = data_object.tokenizer


encoded_input = used_tokenizer.texts_to_sequences([starting_string])[0]


inner_token_list = [encoded_input]
output_generated = ""
iter_thing = 1


inner_input = tf.convert_to_tensor(inner_token_list) 


for i in range(1):
    inner_input = tf.convert_to_tensor([pad_list(inner_token_list[0], 80)])
    predictions  = transformer(inner_input)[:, iter_thing, :]
    print(predictions)
    predicted_id = tf.argmax(predictions[:1], axis=-1)
    
    token = predicted_id.numpy()[0]
    print(token)
    inner_token_list[0].append(token)

    text = used_tokenizer.index_word[token]
    output_generated += (' ' + text)
        
    iter_thing += 1
    inner_token_list[0].append(token)
    

print(output_generated)

