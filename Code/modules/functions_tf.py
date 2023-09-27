import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import BatchNormalization, Layer, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Multiply, Add, Activation, Lambda, Conv1D, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras as keras


def plot_and_save_history(name: str, history, model, save_path: str, subfolder: str, plot_acc=True):
    import matplotlib.pyplot as plt
    import pickle
        
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epochs')
    plt.savefig(f'{save_path}\\plots_data\\loss_{name}.png', dpi=300)
    pickle.dump(history.history, open(f'{save_path}\\plots_data\\history_{name}.pkl', 'wb'))
    if plot_acc:
        plt.figure()
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.legend(['top_categorical_accuracy', 'val_top_categorical_accuracy'])
        plt.xlabel('Epochs')
        plt.ylabel('Categorical Accuracy')
        plt.ylim(0,1)
        plt.savefig(f'{save_path}\\plots_data\\acc_{name}.png', dpi=300)

    model.save(f'{save_path}\\{subfolder}\\{name}.h5')
    return None

class ViTEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.patch_embeddings = tf.keras.layers.Conv1D(filters=hidden_size, kernel_size=patch_size, strides=patch_size)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.cls_token = self.add_weight(shape=(1, 1, self.hidden_size), trainable=True, name="cls_token")

        num_patches = input_shape[1] // self.patch_size
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.hidden_size), trainable=True, name="position_embeddings"
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs_shape = tf.shape(inputs)  # N,H,W,C
        embeddings = self.patch_embeddings(inputs, training=training)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = tf.repeat(self.cls_token, repeats=inputs_shape[0], axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

class MLP(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=None, activation="gelu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim)
        self.activation1 = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(input_shape[-1] if self.out_dim is None else self.out_dim)

    def call(self, inputs: tf.Tensor, training: bool = False):
        x = self.dense1(inputs)
        x = self.activation1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x

class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        attention_dim,
        attention_bias,
        mlp_dim,
        # shapes_output,
        attention_dropout=0.0,
        sd_survival_probability=1.0,
        activation="gelu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_before = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads,
            attention_dim // num_heads,
            use_bias=attention_bias,
            dropout=attention_dropout,
            # output_shape= shapes_output
        )
        self.stochastic_depth = tfa.layers.StochasticDepth(sd_survival_probability)
        self.norm_after = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(mlp_dim=mlp_dim, activation=activation, dropout=dropout)

    def build(self, input_shape):
        super().build(input_shape)
        self.attn._build_from_signature(input_shape, input_shape)

    def call(self, inputs, training=False):
        x = self.norm_before(inputs, training=training)
        x = self.attn(x, x, training=training)
        x = self.stochastic_depth([inputs, x], training=training)
        x2 = self.norm_after(x, training=training)
        x2 = self.mlp(x2, training=training)
        return self.stochastic_depth([x, x2], training=training)

    def get_attention_scores(self, inputs):
        x = self.norm_before(inputs, training=False)
        _, weights = self.attn(x, x, training=False, return_attention_scores=True)
        return weights

class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_dim,
        num_classes,
        # output_shapes,
        dropout=0.0,
        sd_survival_probability=1.0,
        attention_bias=False,
        attention_dropout=0.0,
        output_activation='softmax',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.embeddings = ViTEmbeddings(patch_size, hidden_size, dropout)
        sd = tf.linspace(1.0, sd_survival_probability, depth)
        self.blocks = [
            Block(
                num_heads,
                attention_dim=hidden_size,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                # shapes_output = output_shapes,
                mlp_dim=mlp_dim,
                sd_survival_probability=(sd[i].numpy().item()),
                dropout=dropout,
            )
            for i in range(depth)
        ]

        self.norm = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(num_classes, output_activation)
        

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embeddings(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        x = x[:, 0]  # take only cls_token
        return self.head(x)

    def get_last_selfattention(self, inputs: tf.Tensor):
        x = self.embeddings(inputs, training=False)
        for block in self.blocks[:-1]:
            x = block(x, training=False)
        return self.blocks[-1].get_attention_scores(x)

class ChannelAttention(Layer):
    def __init__(self, filters, ratio, **kwargs):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.shared_layer_one = Dense(self.filters // self.ratio, activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.shared_layer_two = Dense(self.filters, kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')

    def call(self, inputs):
        avg_pool = GlobalAveragePooling1D()(inputs)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling1D()(inputs)
        max_pool = Reshape((1, self.filters))(max_pool)  # Reshape here

        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        attention = Add()([avg_pool, max_pool])
        attention = Activation('sigmoid')(attention)

        return Multiply()([inputs, attention])
    
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'filters': self.filters, 'ratio': self.ratio})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class SpatialAttention(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1d = Conv1D(filters=1, kernel_size=self.kernel_size,
                             strides=1, padding='same', 
                             activation='sigmoid',
                             kernel_initializer='he_normal', 
                             use_bias=False)

    def call(self, inputs):
        avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=2, keepdims=True))(inputs)
        max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=2, keepdims=True))(inputs)

        attention = Concatenate(axis=2)([avg_pool, max_pool])
        attention = self.conv1d(attention)

        return Multiply()([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class CBAM(tf.keras.layers.Layer):
    def __init__(self, num_filters, reduction_ratio=16, **kwargs):
        super(CBAM, self).__init__()
        self.num_filters = num_filters
        self.reduction_ratio = reduction_ratio
        self.channel_attention = ChannelAttention(num_filters, reduction_ratio)
        self.spatial_attention = SpatialAttention(7)

    def call(self, inputs):
        channel_attention = self.channel_attention(inputs)
        spatial_attention = self.spatial_attention(inputs)
        x = tf.keras.layers.Multiply()([inputs, channel_attention])
        x_2 = tf.keras.layers.Multiply()([inputs, spatial_attention])
        x = tf.keras.layers.Add()([x, x_2])
        return x

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({'num_filters': self.num_filters, 'reduction_ratio': self.reduction_ratio})
        return config

def build_1d_resnet_with_cbam(input_shape, num_classes, num_filters=64, res_block_num=3,
                              output_activation='softmax',
                              output_shape=None):
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=7, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = CBAM(num_filters)(x)
    
    # Residual blocks
    for _ in range(res_block_num):
        residual = x
        x = Conv1D(filters=num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters=num_filters, kernel_size=7, padding='same')(x)
        x = BatchNormalization()(x)        
        x = Activation('relu')(x)
        x = Conv1D(filters=num_filters, kernel_size=21, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])
        x = Activation('relu')(x)
        
        x = CBAM(num_filters)(x)
    
    # Global average pooling and output layer
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(num_classes, activation=output_activation)(x)
    outputs = tf.keras.layers.Reshape(output_shape)(x)
    
    model = tf.keras.models.Model(inputs, outputs)
    return model

def custom_loss(y_true, y_pred): 
    # Calculate the squared differences between y_i and y_hat_i
    squared_diff = tf.square(y_true - y_pred)
    
    # Multiply each squared difference by y_i^2
    weighted_squared_diff = tf.multiply(tf.square(y_true), squared_diff)
    
    # Sum the weighted squared differences over all elements
    loss = tf.reduce_sum(weighted_squared_diff)
    
    return loss