def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer and output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Compute the second derivative (Hessian)
    with tf.GradientTape() as tape2:
        tape2.watch(last_conv_layer_output)
        preds = model(img_array)
    second_grads = tape2.gradient(preds[:, pred_index], last_conv_layer_output)

    # Calculate alpha coefficients
    alpha = tf.maximum(grads, 0) / (tf.maximum(grads, 0) + tf.maximum(second_grads, 0))

    # Multiply the alpha coefficients with the feature map output
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.reduce_sum(alpha[..., tf.newaxis] * last_conv_layer_output, axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
