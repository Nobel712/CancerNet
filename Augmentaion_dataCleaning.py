

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(img_path)
    return images, filenames

def remove_duplicate_images(images, filenames):
    unique_images = []
    unique_filenames = []
    seen_hashes = set()
    
    for img, fname in zip(images, filenames):
        img_hash = hash(img.tobytes())
        if img_hash not in seen_hashes:
            seen_hashes.add(img_hash)
            unique_images.append(img)
            unique_filenames.append(fname)
    
    return unique_images, unique_filenames

def detect_outliers(images):
    image_vectors = [cv2.resize(img, (50, 50)).flatten() for img in images]
    scaler = StandardScaler()
    image_vectors_scaled = scaler.fit_transform(image_vectors)
    
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(image_vectors_scaled)
    
    mean_vector = np.mean(reduced_features, axis=0)
    distances = [euclidean(vec, mean_vector) for vec in reduced_features]
    threshold = np.percentile(distances, 95)  # Consider top 5% as outliers
    
    outliers = [images[i] for i in range(len(distances)) if distances[i] > threshold]
    return outliers

def augment_image(image):
    # Brightness adjustment
    factor = random.uniform(0.5, 1.2)
    bright_image = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    # Mirroring
    mirrored_image = cv2.flip(bright_image, 1)
    
    # Rotation
    angle = random.choice([0, 90, 120, 180])
    (h, w) = bright_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(bright_image, rotation_matrix, (w, h))
    
    # Color augmentation (converting to a different color space)
    hematoxylin_cosin_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2YCrCb)
    
    # Zooming
    zoom_factor = 1.2
    zoomed_image = cv2.resize(hematoxylin_cosin_image, None, fx=zoom_factor, fy=zoom_factor)
    
    return zoomed_image

def clean_image_dataset(folder):
    images, filenames = load_images_from_folder(folder)
    images, filenames = remove_duplicate_images(images, filenames)
    outliers = detect_outliers(images)
    
    print(f"Original images: {len(images)}")
    print(f"After duplicate removal: {len(images)}")
    print(f"Outliers detected: {len(outliers)}")
    
    # Apply augmentation
    augmented_images = [augment_image(img) for img in images if img not in outliers]
    
    return augmented_images

# Example usage
folder_path = ""
cleaned_images = clean_image_dataset(folder_path)

#####################################

def load_images(image_folder):
    images = []
    filenames = []
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        try:
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images, filenames

# Remove duplicate images
def remove_duplicates(images, filenames):
    unique_images = []
    unique_filenames = []
    seen_hashes = set()
    
    for img, fname in zip(images, filenames):
        img_hash = hash(img.tobytes())
        if img_hash not in seen_hashes:
            seen_hashes.add(img_hash)
            unique_images.append(img)
            unique_filenames.append(fname)
    
    return unique_images, unique_filenames

# Detect and remove corrupted images
def remove_corrupt_images(images, filenames):
    valid_images = []
    valid_filenames = []
    
    for img, fname in zip(images, filenames):
        if img is not None and img.size > 0:
            valid_images.append(img)
            valid_filenames.append(fname)
    
    return valid_images, valid_filenames

# Detect outliers based on image brightness
def detect_outliers(images, filenames):
    brightness_values = [np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in images]
    Q1, Q3 = np.percentile(brightness_values, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = [(fname, brightness) for fname, brightness in zip(filenames, brightness_values) if brightness < lower_bound or brightness > upper_bound]
    return outliers

# Main execution
def clean_image_dataset(image_folder):
    images, filenames = load_images(image_folder)
    images, filenames = remove_duplicates(images, filenames)
    images, filenames = remove_corrupt_images(images, filenames)
    outliers = detect_outliers(images, filenames)
    
    print("Outliers detected:", outliers)
    return images, filenames
