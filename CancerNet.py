

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

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def CancerNet_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # Convolution Block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Involution Block
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

    # Convolution
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Transformer Block (reshape to sequence)
    h, w = x.shape[1], x.shape[2]
    seq = layers.Reshape((h * w, 128))(x)
    seq = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256)(seq)
    x = layers.Reshape((h, w, 128))(seq)

    # Parallel Path 1: Flatten → Dense → Reshape
    flat_path = layers.Flatten()(x)
    flat_path = layers.Dense(128, activation='relu')(flat_path)
    flat_path = layers.Reshape((8, 8, 2))(flat_path)

    # Parallel Path 2: Activation → Pooling → GAP → Dense → Reshape
    alt_path = layers.Activation('relu')(x)
    alt_path = layers.MaxPooling2D((2, 2))(alt_path)
    alt_path = layers.Activation('relu')(alt_path)
    alt_path = layers.MaxPooling2D((2, 2))(alt_path)
    alt_path = layers.GlobalAveragePooling2D()(alt_path)
    alt_path = layers.Dense(64, activation='relu')(alt_path)
    alt_path = layers.Reshape((8, 8, 1))(alt_path)

    # Merge both paths
    merged = layers.Concatenate(axis=-1)([flat_path, alt_path])

    # Final output
    output = layers.Conv2D(nclasses, (1, 1), activation='sigmoid/softmax')(merged)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CancerNet")





 
