
from operator import ne
import time
import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, kernel_profiler=True, print_ir=False)

chunk_size = 8
width, height = 9*100, 9*100
width = (width//chunk_size)*chunk_size
height = (height//chunk_size)*chunk_size
zoom = 4
print(width, height)

# Create a 640x480 scalar field, each of its elements representing a pixel value (f32)
gray_scale_image = ti.field(dtype=ti.f32, shape=(width, height))
next_step = ti.field(dtype=ti.f32, shape=(width, height))
gray_scale_image2 = ti.field(dtype=ti.f32)
next_step2 = ti.field(dtype=ti.f32)
# ti.root.dense(ti.ij, (width//chunk_size, height//chunk_size)).dense(ti.ij,(chunk_size, chunk_size)).place(gray_scale_image2)
ti.root.dense(ti.ij, (width, height)).place(gray_scale_image2)
ti.root.dense(ti.ij, (width, height)).place(next_step2)

scaled_image = ti.Vector.field(n=1, dtype=ti.f32, shape=(width*zoom, height*zoom))
scaled_image2 = ti.Vector.field(n=1, dtype=ti.f32)
ti.root.dense(ti.ij, (width,height)).dense(ti.ij,(zoom,zoom)).place(scaled_image2)

@ti.kernel
def fill_image(image: ti.template() ,fill_ratio: ti.f32):
    for i, j in image:
        image[i, j] = ti.random() > fill_ratio

@ti.kernel
def game_step(current_state: ti.template(), next_state: ti.template()):
    for i,j in current_state:       
        
        # Remove middle cell from neighbours
        count: ti.f32 = 0
        
        # count neighbors
        count += current_state[(i-1) %width, (j-1)%height]
        count += current_state[i,            (j-1)%height]
        count += current_state[(i+1) %width, (j-1)%height]
        
        count += current_state[(i-1) %width, j]
        count += current_state[(i+1) %width, j]

        count += current_state[(i-1) %width, (j+1)%height]
        count += current_state[i,            (j+1)%height]
        count += current_state[(i+1) %width, (j+1)%height]

        # Apply rules
        if current_state[i,j] == 1.0:
            next_state[i, j] = 1.0
            if count < 2 or count > 3:
                next_state[i, j] = 0.0
        else:
            next_state[i, j] = 0.0
            if count == 3:
                next_state[i, j] = 1.0
                
                
    # Move data to current_state
    for i,j in next_state:
        # Apply step
        if next_state[i,j] == 1.0:
            current_state[i,j] = 1.0
        else:
            current_state[i,j] = 0.0
        
        # next_step[i,j] = 0.0
        
@ti.kernel
def game_step2(current_state: ti.template(), next_state: ti.template()):
    
    for i,j in current_state:       
        ti.block_local(current_state)
        count: ti.f32 = 0.0
        # count neighbors
        ti.loop_config(serialize=True)
        for x,y in ti.static(ti.ndrange((-1,2),(-1,2))):
            if x != 0 or y != 0:
                ti.atomic_add(count, current_state[(i+x) %width, (j+y)%height])
        
        # Apply rules
        if current_state[i,j] == 1.0:
            next_state[i, j] = 1.0
            if count < 2 or count > 3:
                next_state[i, j] = 0.0
        else:
            next_state[i, j] = 0.0
            if count == 3:
                next_state[i, j] = 1.0
                
                
    # Move data to current_state
    for i,j in next_state:
        ti.block_local(next_state)
        # Apply step
        if next_state[i,j] == 1.0:
            current_state[i,j] = 1.0
        else:
            current_state[i,j] = 0.0
        
        # next_step[i,j] = 0.0
        
@ti.kernel
def zoom_image(input_image: ti.template(), output_image: ti.template()):
    zoom_factor = ti.static(zoom)
    for i,j in output_image:
        output_image[i,j][0] = input_image[i//zoom_factor, j//zoom_factor]

@ti.kernel
def set_pixel(x: ti.i32, y: ti.i32, value: ti.f32):
    gray_scale_image[x, y] = value
    


def main():
    window = ti.ui.Window('Game of Life', (width*zoom, height*zoom), vsync=False)
    canvas = window.get_canvas()
    
    # GUI Variables
    paused = False
    step_size = 1
    fill_ratio = 0.5
    fps = 60.0
    
    fill_image(gray_scale_image, fill_ratio)
    
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
            zoom_image(gray_scale_image, scaled_image)
        
        # Gui Logic
        dt = 1.0/fps
        if reset_clicked:
            fill_image(gray_scale_image, fill_ratio)
            zoom_image(gray_scale_image, scaled_image)
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
                    game_step(gray_scale_image, next_step)
                    zoom_image(gray_scale_image, scaled_image)
        canvas.set_image(scaled_image)
        window.show()
        
        
@ti.kernel
def copy_field(a: ti.template(), b: ti.template()):
    for I in ti.grouped(a):
        b[I] = a[I]

def profile():
    fill_image(gray_scale_image, 0.5)
    copy_field(gray_scale_image, gray_scale_image2)
    ti.profiler.clear_kernel_profiler_info()  # clear all records 
    for i in range(10):
        game_step(gray_scale_image, next_step)
        game_step2(gray_scale_image2, next_step2)
        # zoom_image(gray_scale_image, scaled_image)
    
    ti.profiler.print_kernel_profiler_info()
    ti.profiler.clear_kernel_profiler_info()  # clear all records    
        
    assert(np.allclose(gray_scale_image.to_numpy() , gray_scale_image2.to_numpy()))

if __name__ == "__main__":
    # main()
    profile()