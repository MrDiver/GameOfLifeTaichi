
import taichi as ti
ti.init(arch=ti.gpu)

width, height = 3840, 2160
zoom = 1

# Create a 640x480 scalar field, each of its elements representing a pixel value (f32)
gray_scale_image = ti.field(dtype=ti.f32, shape=(width, height))
next_step = ti.field(dtype=ti.f32, shape=(width, height))
scaled_image = ti.Vector.field(n=1, dtype=ti.f32, shape=(width*zoom, height*zoom))

@ti.kernel
def fill_image():
    for i, j in gray_scale_image:
        gray_scale_image[i, j] = ti.random() > 0.5

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

        # Apply rules
        if gray_scale_image[i,j] == 1.0:
            next_step[i, j] = 1.0
            if count < 2 or count > 3:
                next_step[i, j] = 0.0
        else:
            next_step[i, j] = 0.0
            if count == 3:
                next_step[i, j] = 1.0
                
                
    # Move data to gray_scale_image
    for i,j in next_step:
        # Apply step
        if next_step[i,j] == 1.0:
            gray_scale_image[i,j] = 1.0
        else:
            gray_scale_image[i,j] = 0.0
        
        # next_step[i,j] = 0.0
        
@ti.kernel
def zoom_image():
    zoom_factor = ti.static(zoom)
    for i,j in gray_scale_image:
        for x,y in ti.ndrange((0,zoom_factor), (0,zoom_factor)):
            scaled_image[i*zoom_factor+x, j*zoom_factor+y][0] = gray_scale_image[i,j]
            
@ti.kernel
def zoom_image2():
    zoom_factor = ti.static(zoom)
    for i,j in scaled_image:
        scaled_image[i,j][0] = gray_scale_image[i//zoom_factor, j//zoom_factor]



if __name__ == "__main__":
    window = ti.ui.Window('Game of Life', (width*zoom, height*zoom))
    canvas = window.get_canvas()
    fill_image()
    while window.running:
        count_neighbors_and_apply()
        zoom_image2()
        canvas.set_image(scaled_image)
        window.show()