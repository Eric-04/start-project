import numpy as np

def fragment_texture(image_size):
    # Generate fragmented texture
    texture = np.zeros((image_size, image_size))

    # Add random fragmented shapes
    num_shapes = 10
    min_shape_size = 2
    max_shape_size = 10
    for _ in range(num_shapes):
        shape_size = np.random.randint(min_shape_size, max_shape_size)
        x = np.random.randint(0, image_size - shape_size)
        y = np.random.randint(0, image_size - shape_size)
        texture[x:x+shape_size, y:y+shape_size] = 1
    return texture

def spiral_texture(image_size):
    # Generate spiral texture
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Create spiral pattern
    texture = np.sin(radius * 20 - theta * 10)
    return texture