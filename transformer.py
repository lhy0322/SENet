import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


# class TransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
#         super(TransformerBlock, self).__init__()
#         self.att = MultiHeadSelfAttention(embed_dim, num_heads)
#         self.mlp = tf.keras.Sequential(
#             [
#                 Dense(mlp_dim, activation=tfa.activations.gelu),
#                 Dropout(dropout),
#                 Dense(embed_dim),
#                 Dropout(dropout),
#             ]
#         )
#         self.layernorm1 = LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = LayerNormalization(epsilon=1e-6)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#
#     def call(self, inputs, training):
#         inputs_norm = self.layernorm1(inputs)
#         attn_output = self.att(inputs_norm)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = attn_output + inputs
#
#         out1_norm = self.layernorm2(out1)
#         mlp_output = self.mlp(out1_norm)
#         mlp_output = self.dropout2(mlp_output, training=training)
#         return mlp_output + out1