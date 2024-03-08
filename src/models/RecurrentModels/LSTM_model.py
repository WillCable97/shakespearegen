import tensorflow as tf

def LSTM_model(vocab_size:int, embedding_dim: int, rnn_units:int, batch_size:int) -> tf.keras.models.Model: #Batch size is nessecary as the model is stateful
    stateful_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]), 
        tf.keras.layers.LSTM(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return stateful_model