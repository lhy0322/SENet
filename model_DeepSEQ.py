import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import transformer
import numpy as np
import sonnet as snt
import tensorflow as tf

NB_WORDS = 4097
EMBEDDING_DIM = 100
embedding_matrix = np.load('embedding_matrix.npy')


class Enformer(snt.Module):
  """Main model."""

  def __init__(self,
               channels: int = 128,
               num_transformer_layers: int = 4,
               dropout_rate: float = 0.2,
               num_heads: int = 8,
               name: str = 'enformer'):
    """Enformer model.

    Args:
      channels: Number of convolutional filters and the overall 'width' of the
        model.
      num_transformer_layers: Number of transformer layers.
      num_heads: Number of attention heads.
      name: Name of sonnet module.
    """
    super().__init__(name=name)

    trunk_name_scope = tf.name_scope('trunk')
    trunk_name_scope.__enter__()

    # Transformer.
    def transformer_mlp():
      return Sequential(lambda: [
          snt.LayerNorm(axis=-1, create_scale=True, create_offset=True),
          snt.Linear(channels * 2),
          snt.Dropout(dropout_rate),
          tf.nn.relu,
          snt.Linear(channels),
          snt.Dropout(dropout_rate)], name='mlp')

    self.transformer = Sequential(lambda: [
        Sequential(lambda: [
            Residual(Sequential(lambda: [
                snt.LayerNorm(axis=-1,
                              create_scale=True, create_offset=True,
                              scale_init=snt.initializers.Ones()),
                transformer.MultiHeadSelfAttention(embed_dim=channels, num_heads=num_heads),

                snt.Dropout(dropout_rate)], name='mha')),
            Residual(transformer_mlp())], name=f'transformer_block_{i}')
        for i in range(num_transformer_layers)], name='transformer')

    self.emb = Sequential(lambda: [
        # snt.Embed(NB_WORDS, EMBEDDING_DIM),
        snt.Embed(existing_vocab=embedding_matrix),
    ], name='emb')

    self.cnn = Sequential(lambda: [
        # snt.Embed(NB_WORDS, EMBEDDING_DIM),
        # snt.Embed(existing_vocab=embedding_matrix),
        # snt.Conv1D(channels, 40, padding='SAME'),
        # snt.BatchNorm(create_scale=True,
        #               create_offset=True,
        #               decay_rate=0.9,
        #               scale_init=snt.initializers.Ones()),
        # tf.nn.relu,
        # SoftmaxPooling1D(20),
        # -----------------------------------------------------
        snt.Conv1D(channels, 40, padding='SAME'),
        snt.BatchNorm(create_scale=True,
                      create_offset=True,
                      decay_rate=0.9,
                      scale_init=snt.initializers.Ones()),
        tf.nn.relu,
        SoftmaxPooling1D(5),
        snt.Conv1D(channels, 40, padding='SAME'),
        snt.BatchNorm(create_scale=True,
                      create_offset=True,
                      decay_rate=0.9,
                      scale_init=snt.initializers.Ones()),
        tf.nn.relu,
        SoftmaxPooling1D(4),
        # -----------------------------------------------------
        # snt.Conv1D(channels, 40, padding='SAME'),
        # snt.BatchNorm(create_scale=True,
        #               create_offset=True,
        #               decay_rate=0.9,
        #               scale_init=snt.initializers.Ones()),
        # tf.nn.relu,

        # snt.Conv1D(channels, 40, padding='SAME'),
        # snt.BatchNorm(create_scale=True,
        #               create_offset=True,
        #               decay_rate=0.9,
        #               scale_init=snt.initializers.Ones()),
        # tf.nn.relu,

        # snt.Conv1D(channels, 40, padding='SAME'),
        # snt.BatchNorm(create_scale=True,
        #               create_offset=True,
        #               decay_rate=0.9,
        #               scale_init=snt.initializers.Ones()),
        # tf.nn.relu,
        # SoftmaxPooling1D(2),
        # tf.keras.layers.MaxPool1D(20, 20, padding='VALID'),
        # tf.keras.layers.AveragePooling1D(20, 20, padding='VALID'),
    ], name='cnn')

    self.mlp = Sequential(lambda: [
        # snt.Linear(128),

        # snt.BatchNorm(create_scale=True,
        #               create_offset=True,
        #               decay_rate=0.9,
        #               scale_init=snt.initializers.Ones()),
        # tf.nn.relu,
        snt.Linear(64),
        snt.BatchNorm(create_scale=True,
                      create_offset=True,
                      decay_rate=0.9,
                      scale_init=snt.initializers.Ones()),
        tf.nn.relu,
        snt.Dropout(dropout_rate),
        snt.Linear(1)

    ], name='mlp')


    trunk_name_scope.__exit__(None, None, None)

  @tf.function
  def __call__(self, inputs: tf.Tensor, is_training: bool) -> Dict[str, tf.Tensor]:

    x = self.emb(inputs, is_training=is_training)
    x = self.cnn(x, is_training=is_training)
    x = self.transformer(x, is_training=is_training)
    x = tf.keras.layers.Flatten()(x)
    x = self.mlp(x, is_training=is_training)
    return x

class Sequential(snt.Module):
  """snt.Sequential automatically passing is_training where it exists."""

  def __init__(self,
               layers: Optional[Union[Callable[[], Iterable[snt.Module]],
                                      Iterable[Callable[..., Any]]]] = None,
               name: Optional[Text] = None):
    super().__init__(name=name)
    if layers is None:
      self._layers = []
    else:
      # layers wrapped in a lambda function to have a common namespace.
      if hasattr(layers, '__call__'):
        with tf.name_scope(name):
          layers = layers()
      self._layers = [layer for layer in layers if layer is not None]

  def __call__(self, inputs: tf.Tensor, is_training: bool, **kwargs):
    outputs = inputs
    for _, mod in enumerate(self._layers):
      if accepts_is_training(mod):
        outputs = mod(outputs, is_training=is_training, **kwargs)
      else:
        outputs = mod(outputs, **kwargs)
    return outputs


class SoftmaxPooling1D(snt.Module):
  """Pooling operation with optional weights."""

  def __init__(self,
               pool_size: int = 2,
               per_channel: bool = False,
               w_init_scale: float = 0.0,
               name: str = 'softmax_pooling'):
    """Softmax pooling.

    Args:
      pool_size: Pooling size, same as in Max/AvgPooling.
      per_channel: If True, the logits/softmax weights will be computed for
        each channel separately. If False, same weights will be used across all
        channels.
      w_init_scale: When 0.0 is equivalent to avg pooling, and when
        ~2.0 and `per_channel=False` it's equivalent to max pooling.
      name: Module name.
    """
    super().__init__(name=name)
    self._pool_size = pool_size
    self._per_channel = per_channel
    self._w_init_scale = w_init_scale
    self._logit_linear = None

  @snt.once
  def _initialize(self, num_features):
    self._logit_linear = snt.Linear(
        output_size=num_features if self._per_channel else 1,
        with_bias=False,  # Softmax is agnostic to shifts.
        w_init=snt.initializers.Identity(self._w_init_scale))

  def __call__(self, inputs):
    _, length, num_features = inputs.shape
    self._initialize(num_features)
    inputs = tf.reshape(
        inputs,
        (-1, length // self._pool_size, self._pool_size, num_features))
    return tf.reduce_sum(
        inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2),
        axis=-2)


class Residual(snt.Module):
  """Residual block."""

  def __init__(self, module: snt.Module, name='residual'):
    super().__init__(name=name)
    self._module = module

  def __call__(self, inputs: tf.Tensor, is_training: bool, *args,
               **kwargs) -> tf.Tensor:
    return inputs + self._module(inputs, is_training, *args, **kwargs)


def accepts_is_training(module):
  return 'is_training' in list(inspect.signature(module.__call__).parameters)
