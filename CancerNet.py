

# Define the custom Involution layer
class Involution(keras.layers.Layer):
    def __init__(
        self, channel, group_number, kernel_size, stride, reduction_ratio, name
    ):
        super().__init__(name=name)
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        (_, height, width, num_channels) = input_shape
        height = height // self.stride
        width = width // self.stride

        self.stride_layer = (
            keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )
        self.kernel_gen = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        self.kernel_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )

    def call(self, x):
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)
        kernel = self.kernel_reshape(kernel)

        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        input_patches = self.input_patches_reshape(input_patches)

        output = tf.multiply(kernel, input_patches)
        output = tf.reduce_sum(output, axis=3)
        output = self.output_reshape(output)

        return output, kernel

# Build the advanced model for binary classification with (None, 1) output
inputs = keras.Input(shape=(256, 256, 3))
x = inputs

# Add CNN blocks
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)

# Add Involution blocks
x, kernel1 = Involution(
    channel=64, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_1")(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)

x, kernel2 = Involution(
    channel=64, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_2")(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)

x, kernel3 = Involution(
    channel=64, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_3")(x)
x = keras.layers.ReLU()(x)

# Flatten and add Dense layers
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation="relu")(x)

# Reshape x to (batch_size, sequence_length, embedding_dimension)
# for MultiHeadAttention input
x = keras.layers.Reshape((-1, 64))(x)  # sequence_length is dynamic based on flatten output

# Incorporate Transformer blocks (Self-Attention mechanism)
num_heads = 4
ff_dim = 64
num_transformer_blocks = 2

for i in range(num_transformer_blocks):
    # Multi-head self-attention
    attention = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=64, dropout=0.1
    )(x, x)
    attention = keras.layers.LayerNormalization(epsilon=1e-6)(attention + x)

    # Feed-forward neural network
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(attention)
    x = keras.layers.Dense(units=64, activation="relu")(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attention)

# Apply Global Average Pooling to collapse sequence dimension
x = keras.layers.GlobalAveragePooling1D()(x)

# Output layer for binary classification with (None, 1) shape
outputs = keras.layers.Dense(3, activation="softmax")(x)

# Define and compile the model
model = keras.Model(inputs=inputs, outputs=outputs, name="CancerNet")


####################################################################################



class Involution(layers.Layer):
    """ Involution Layer: Inspired by the 'Involution: Inverting the Inherence of Convolution' paper. """
    def __init__(self, channels, kernel_size=3, stride=1):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.conv1 = layers.Conv2D(channels, 1)  # 1x1 Conv for Channel Mixing
        self.conv2 = layers.Conv2D(kernel_size ** 2, 1)  # Kernel Generator

    def call(self, x):
        batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        weights = self.conv2(x)  
        weights = tf.reshape(weights, [batch_size, height, width, self.kernel_size, self.kernel_size, 1])
        x_unfold = tf.image.extract_patches(x, [1, self.kernel_size, self.kernel_size, 1], 
                                            [1, self.stride, self.stride, 1], [1, 1, 1, 1], padding="SAME")
        x_unfold = tf.reshape(x_unfold, [batch_size, height, width, self.kernel_size, self.kernel_size, channels])
        x = tf.reduce_sum(weights * x_unfold, axis=[3, 4])
        x = self.conv1(x)
        return x


class TransformerBlock(layers.Layer):
    """ Transformer Block: Implements a self-attention mechanism followed by an MLP block. """
    def __init__(self, dim, num_heads=4, mlp_dim=256, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=dropout)
        self.norm2 = layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='relu'),
            layers.Dense(dim)
        ])

    def call(self, x):
        attn_out = self.attn(x, x)
        x = self.norm1(x + attn_out)  # Residual Connection
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)  # Residual Connection
        return x


def build_model(input_shape=(224, 224, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # **Step 1: Convolution → Activation**
    x = layers.Conv2D(64, (3, 3), padding="same")(inputs)
    x = layers.ReLU()(x)

    # **Step 2: MaxPool → Convolution → Activation**
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.ReLU()(x)

    # **Step 3: MaxPool → Involution → Transformer**
    x = layers.MaxPooling2D((2, 2))(x)
    x = Involution(128)(x)

    # **Step 4: Transformer Block**
    x = layers.Reshape((-1, 128))(x)  # Reshape for Transformer
    x = TransformerBlock(dim=128)(x)

    # **Step 5: Global Average Pooling → Fully Connected → Flatten**
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Flatten()(x)

    # **Step 6: Fully Connected → Output**
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)




 