import numpy as np
from skimage.transform import AffineTransform, warp

# Parameters
num_samples_per_class = 100
image_size = 28
textures = ['smooth', 'rough', 'checkerboard']

# Function to generate textures
def generate_texture(texture_type):
    if texture_type == 'rough':
        return np.random.rand(image_size, image_size)
    elif texture_type == 'smooth':
        return np.tile(np.linspace(0, 1, image_size), (image_size, 1))
    elif texture_type == 'checkerboard':
        checkerboard = np.indices((image_size, image_size)).sum(axis=0) % 2
        return checkerboard
    
def apply_random_transformation(image):
    rotation_angle = np.deg2rad(np.random.randint(-45, 45))
    scale_factor = np.random.uniform(0.5, 1.5)

    # Define the affine transformation matrix
    matrix = np.array([[np.cos(rotation_angle) * scale_factor, -np.sin(rotation_angle), 0],
                    [np.sin(rotation_angle), np.cos(rotation_angle) * scale_factor, 0],
                    [0, 0, 1]])

    # Create an AffineTransform object
    affine_transform = AffineTransform(matrix=matrix)

    # Apply the affine transformation to the image
    transformed_image = warp(image, affine_transform)

    return transformed_image

def generate_data():
    # Generate dataset
    dataset = []
    labels = []

    for _ in range(3):
        for texture in textures:
            for _ in range(num_samples_per_class):
                image = generate_texture(texture)
                image = apply_random_transformation(image)
                dataset.append(image)
                labels.append((texture))

    return dataset, labels