
def load_images_from_folder(folder):
    
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None and img.size > 0:
                images.append(img)
                filenames.append(img_path)
            else:
                print(f"Skipped invalid or empty image: {filename}")
        except Exception as e:
            print(f"Error reading image {filename}: {e}")
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
        else:
            print(f"Duplicate image skipped: {fname}")
    
    return unique_images, unique_filenames

def augment_image(image):
 
    # Random brightness
    factor = random.uniform(0.7, 1.3)
    bright_image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Horizontal flip
    mirrored_image = cv2.flip(bright_image, 1)

    # Random rotation
    angle = random.choice([0, 90, 120, 180])
    (h, w) = mirrored_image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated_image = cv2.warpAffine(mirrored_image, matrix, (w, h))

    # Convert color space (simulate stain variation)
    color_augmented = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2YCrCb)

    # Zoom (scale and then crop to original size)
    zoom_factor = 1.2
    zoomed = cv2.resize(color_augmented, None, fx=zoom_factor, fy=zoom_factor)
    zh, zw = zoomed.shape[:2]
    startx = (zw - w) // 2
    starty = (zh - h) // 2
    zoomed_cropped = zoomed[starty:starty + h, startx:startx + w]

    return zoomed_cropped

def clean_image_dataset(folder, augment=False):

    print("Loading images...")
    images, filenames = load_images_from_folder(folder)
    print(f"Loaded {len(images)} images.")

    print("Removing duplicates...")
    images, filenames = remove_duplicate_images(images, filenames)
    print(f"After removing duplicates: {len(images)} images remain.")

    if augment:
        print("Applying data augmentation...")
        images = [augment_image(img) for img in images]

    return images, filenames

# Example usage
if __name__ == "__main__":
    #folder_path = "/path"
    #cleaned_images, cleaned_filenames = clean_image_dataset(folder_path, augment=True)
