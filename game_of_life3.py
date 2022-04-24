
import time
from matplotlib import image
from matplotlib.pyplot import fill
import taichi as ti
ti.init(arch=ti.gpu)

width, height = 320, 240
zoom = 6

# Create a 640x480 scalar field, each of its elements representing a pixel value (f32)
gray_scale_image = ti.field(dtype=ti.f32, shape=(width, height))
next_step = ti.field(dtype=ti.f32, shape=(width, height))
scaled_image = ti.Vector.field(n=1, dtype=ti.f32, shape=(width*zoom, height*zoom))

@ti.kernel
def fill_image(fill_ratio: ti.f32):
    for i, j in gray_scale_image:
        gray_scale_image[i, j] = ti.random() > fill_ratio

@ti.kernel
def count_neighbors_and_apply():
    for i,j in gray_scale_image:       
        
        # Remove middle cell from neighbours
        count: ti.f32 = 0
        
        # count neighbors
        count += gray_scale_image[(i-1) %width, (j-1)%height]
        count += gray_scale_image[i,            (j-1)%height]
        count += gray_scale_image[(i+1) %width, (j-1)%height]
        
        count += gray_scale_image[(i-1)%width, j]
        count += gray_scale_image[(i+1)%width, j]

        count += gray_scale_image[(i-1) %width, (j+1)%height]
        count += gray_scale_image[i,            (j+1)%height]
        count += gray_scale_image[(i+1) %width, (j+1)%height]

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

@ti.kernel
def set_pixel(x: ti.i32, y: ti.i32, value: ti.f32):
    gray_scale_image[x, y] = value

if __name__ == "__main__":
    window = ti.ui.Window('Game of Life', (width*zoom, height*zoom), vsync=False)
    canvas = window.get_canvas()
    
    # GUI Variables
    paused = False
    step_size = 1
    fill_ratio = 0.5
    fps = 60.0
    
    fill_image(fill_ratio)
    
    # delta time
    timer_start = time.time()
    while window.running:
        # draw gui
        window.GUI.begin("Control Panel", 0, 0.8, 1, 0.2)
        reset_clicked = window.GUI.button("Reset!")
        fill_ratio = window.GUI.slider_float("Fill Ratio",fill_ratio,0,1)
        pause_clicked = window.GUI.button("Pause!")
        play_clicked = window.GUI.button("Play!")
        step_size = int(window.GUI.slider_float("Step",step_size,1,1000))
        fps = int(window.GUI.slider_float("FPS",fps,1,144))
        window.GUI.end()        
        
        # Window Events
        # events = window.get_events()
        # mouse event processing
        mouse = window.get_cursor_pos()
        # ...
        if window.is_pressed(ti.ui.LMB):
            # print(mouse)
            gray_scale_image[int(mouse[0]*width), int(mouse[1]*height)] = 1.0
            zoom_image2()
        
        # Gui Logic
        dt = 1.0/fps
        if reset_clicked:
            fill_image(fill_ratio)
            zoom_image2()
        if pause_clicked:
            paused = True
        if play_clicked:
            paused = False
        
        # Processing
        if not paused:
            diff = time.time() - timer_start
            if diff > dt:
                timer_start = time.time()
                for _ in range(step_size):
                    count_neighbors_and_apply()
                    zoom_image2()
        canvas.set_image(scaled_image)
        window.show()