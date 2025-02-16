import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


# Function to preprocess input image
def preprocess_img(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    return np.array(img), img_array  # Return original image as a NumPy array

# Function to explain a prediction using LIME
def explain_with_lime(img_path, model, num_samples=1000, num_features=10, target_class=None):
    # Preprocess the image
    original_img, img_array = preprocess_img(img_path)
    
    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img_array[0],
        model.predict,
        top_labels=5,
        hide_color=0,  # Transparent hide color to retain the image background
        num_samples=num_samples
    )
    
    # Determine target class for explanation
    if target_class is None:
        target_class = np.argmax(model.predict(img_array))
    
    # Get the explanation mask
    temp, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=True,  # Highlight positive contributions
        num_features=num_features,
        hide_rest=False
    )
    
    # Ensure dimensions match
    overlay = temp / 255.0  # Normalize temp to range [0, 1]
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Make mask 3D (same as RGB image)
    
    # Apply the mask to the original image
    highlighted_img = np.copy(original_img / 255.0)  # Normalize original image
    highlighted_img[mask] = overlay[mask]  # Apply explanation overlay

    # Plot original and explanation images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img.astype(np.uint8))  # Convert back to uint8 for display
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("LIME Explanation")
    plt.imshow(mark_boundaries(highlighted_img, mask[:, :, 0]))
    plt.axis("off")
    plt.tight_layout()
    plt.show()
