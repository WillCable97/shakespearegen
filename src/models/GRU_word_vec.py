import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization



def GRU_word_vec(vocab_size:int, embedding_dim: int, rnn_units:int, batch_size:int):
    vectorizer = TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=20)
    vectorizer.adapt(sentences) 


    # Define the LSTM model
    model = Sequential([
        TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=20),
        Embedding(input_dim=1000, output_dim=50, input_length=20),
        LSTM(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'),  # LSTM layer with 64 units
        Dense(1, activation='sigmoid')  # Binary classification example
    ])

        tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]), 
        tf.keras.layers.LSTM(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
"""
# Example sentences
sentences = ["This is an example sentence.", "Another sentence for illustration."]

# Create a TextVectorization layer
vectorizer = TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=20)
vectorizer.adapt(sentences)

# Define the LSTM model
model = Sequential([
    TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=20),
    Embedding(input_dim=1000, output_dim=50, input_length=20),
    LSTM(64),  # LSTM layer with 64 units
    Dense(1, activation='sigmoid')  # Binary classification example
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the model
model.summary()
"""















import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Example sentences
sentences = ["This is an example sentence.", "Another sentence for illustration."]

# Create a TextVectorization layer
vectorizer = TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=20)
vectorizer.adapt(sentences)

# Define the LSTM model
model = Sequential([
    TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=20),
    Embedding(input_dim=1000, output_dim=50, input_length=20),
    LSTM(64),  # LSTM layer with 64 units
    Dense(1, activation='sigmoid')  # Binary classification example
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the model
model.summary()













