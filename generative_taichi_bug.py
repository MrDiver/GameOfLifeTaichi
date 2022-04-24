import time
import taichi as ti
ti.init(arch=ti.cpu)

width, height = 640,480
# Create a 640x480 scalar field, each of its elements representing a pixel value (f32)
gray_scale_image = ti.field(dtype=ti.f32, shape=(width, height))
next_step = ti.field(dtype=ti.f32, shape=(width, height))
# buffer = ti.field(dtype=ti.f32, shape=(width-1, height))

# @ti.kernel
# def move_image():
#     # Fill the image with random gray
#     for i, j in buffer:
#         buffer[i, j] = gray_scale_image[i, j]
#     for j in range(height):
#         gray_scale_image[0, j] = ti.random()
#     for i,j in buffer:
#         gray_scale_image[i+1, j] = buffer[i, j]

@ti.kernel
def fill_image():
    for i, j in gray_scale_image:
        gray_scale_image[i, j] = ti.random() > 0.1

# fill beacon
@ti.kernel
def fill_beacon():
    gray_scale_image[1,1] = 1
    gray_scale_image[2,1] = 1
    gray_scale_image[1,2] = 1
    gray_scale_image[2,2] = 1
    gray_scale_image[3,3] = 1
    gray_scale_image[3,4] = 1
    gray_scale_image[4,3] = 1
    gray_scale_image[4,4] = 1
            

@ti.kernel
def count_neighbors_and_apply():
    for i,j in ti.ndrange((1, width-1), (1, height-1)):       
        
        # Remove middle cell from neighbours
        count: ti.f32 = 0
        
        # count neighbors
        count += gray_scale_image[i-1, j-1]
        count += gray_scale_image[i, j-1]
        count += gray_scale_image[i+1, j-1]
        
        count += gray_scale_image[i-1, j]
        count += gray_scale_image[i+1, j]

        count += gray_scale_image[i-1, j+1]
        count += gray_scale_image[i, j+1]
        count += gray_scale_image[i+1, j+1]

        # next_step[i,j] = count
        
        # Apply rules
        if gray_scale_image == 1:
            if count < 2 or count > 3:
                next_step[i, j] = 0.0
        else:
            if count == 3:
                next_step[i, j] = 1.0
        
                
    # Move data to gray_scale_image
    for i,j in next_step:
        # Apply step
        if next_step[i,j] == 1:
            gray_scale_image[i,j] = 1.0
        else:
            gray_scale_image[i,j] = 0.0

# Create a GUI of same size as the gray-scale image
gui = ti.GUI('gray-scale image with random values', (width, height))

fill_image()
gui.set_image(gray_scale_image)
gui.show()

while gui.running:
    count_neighbors_and_apply()
    gui.set_image(gray_scale_image)
    gui.show()