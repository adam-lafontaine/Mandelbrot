# Mandelbrot Renderer

## Features
* Explore the mandelbrot set using keyboard controls
* Controller input available
* Select color scheme
* Color contrast maximized for the current visible section


### Keyboard Controls
* Pan up, left, down, right with W, A, S, D or 8, 4, 2, 6 (numpad)
* Zoom in with '+' (numpad)
* Zoom out with '-' (numpad)
* Increase zoom rate with '*'
* Decrease zoom rate with '/'
* Increase resolution with up arrow
* Decrease resolution with down arrow
* Change color scheme with left and right arrow
* End program with ESC key


### Controller Controls (Xinput, SDL2)
* Pan up, left, down, right with right thumbstick
* Zoom in and out with left thumbstick
* Increase zoom rate with right trigger
* Decrease zoom rate with left trigger
* Increase resolution with D pad up
* Decrease resolution with D pad down
* Cycle color scheme with D pad left and right
* End program with B button


### Windows native
* Compile and run in Visual Studio
* No third party libraries to install
* No controller support


### Windows with SDL2
 * Install SDL2 - e.g. .\vcpkg.exe install sdl2:x64-windows
 * Compile and run in Visual Studio


### Ubuntu Linux with SDL2
* Install SDL2 - sudo apt-get install libsdl2-dev
* Makefile in /src/sdl/
* Create build directory - make setup
* Run program - make run


### Windows with SDL2 and CUDA
 * Install SDL2 - e.g. .\vcpkg.exe install sdl2:x64-windows
 * Install CUDA Toolkit - https://developer.nvidia.com/cuda-downloads
 * Compile and run in Visual Studio


### Ubuntu Linux with SDL2 and CUDA
* Install SDL2 - sudo apt-get install libsdl2-dev
* Install CUDA Toolkit - https://developer.nvidia.com/cuda-downloads
* Makefile in /src/sdl_cuda/
    * Update the CUDA_PATH variable at the top of the file
* Create build directory - make setup
* Run program - make run


### Jetson Nano with SDL2
* Install SDL2 - sudo apt-get install libsdl2-dev
* SDL bug with ARM processors - sudo killall ibus-daemon
* Makefile in /src/sdl_nano/
    * Update the CUDA_PATH variable at the top of the file
* Create build directory - make setup
* Run program - make run