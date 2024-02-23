import tensorflow as tf

class DenseComponent(tf.keras.layers.Layer):
    """
        Feed forward componenet implemented after each attention block
        Takes on size of dense dimension (initial layer)
        Will output original embedding dimension
    """
    def __init__(self, embedding_dimension, dense_dimension, dropout_rate=0.1):
        super().__init__()
        #For som fkn reason sequential breaks the whole thing ??
        self.d1 = tf.keras.layers.Dense(dense_dimension, activation='relu')
        self.d2 = tf.keras.layers.Dense(embedding_dimension)
        self.d3 = tf.keras.layers.Dropout(dropout_rate)
        
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x_1 = self.d1(x)
        x_1 = self.d2(x_1)
        x_1 = self.d3(x_1)
        x = self.add([x, x_1])
        x = self.layer_norm(x)
        return x

