import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp

# Parameters
num_samples_per_class = 100
image_size = 28
shapes = ['square', 'circle', 'triangle']
textures = ['smooth', 'rough', 'checkerboard']

# Function to generate shapes
def generate_shape(shape_type):
    if shape_type == 'square':
        return np.ones((image_size, image_size))
    elif shape_type == 'circle':
        x, y = np.indices((image_size, image_size))
        return ((x - image_size/2)**2 + (y - image_size/2)**2) <= (image_size/2)**2
    elif shape_type == 'triangle':
        triangle = np.zeros((image_size, image_size))
        triangle[-1, :] = 1
        for i in range(image_size):
            triangle[i, i:] = 1
        return triangle

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
    matrix = np.array([[np.cos(rotation_angle) * scale_factor, -np.sin(rotation_angle), -image_size*0.25],
                    [np.sin(rotation_angle), np.cos(rotation_angle) * scale_factor, -image_size*0.25],
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

    for shape in shapes:
        for texture in textures:
            for _ in range(num_samples_per_class):
                image = generate_shape(shape) * generate_texture(texture)
                image = apply_random_transformation(image)
                dataset.append(image)
                labels.append((shape, texture))

    # Convert lists to arrays
    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels

# dataset, labels = generate_data()
# # Visualize a few samples
# fig, axes = plt.subplots(len(shapes), len(textures), figsize=(10, 10))

# for i, shape in enumerate(shapes):
#     for j, texture in enumerate(textures):
#         index = np.where((labels[:, 0] == shape) & (labels[:, 1] == texture))[0][0]
#         axes[i, j].imshow(dataset[index], cmap='gray')
#         axes[i, j].set_title(f'{shape} - {texture}')
#         axes[i, j].axis('off')

# plt.tight_layout()
# plt.show()

