# Mandelbrot Renderer

## Features
* Explore the mandelbrot set using keyboard controls
* Controller input available for Linux
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


### Controller Controls (Xinput, Linux only)
* Pan up, left, down, right with right thumbstick
* Zoom in and out with left thumbstick
* Increase zoom rate with right trigger
* Decrease zoom rate with left trigger
* Increase resolution with D pad up
* Decrease resolution with D pad down
* Cycle color scheme with D pad left and right
* End program with B button


### Visual Studio solution to run on Windows
* Compile and run
* No third party libraries to install
* No controller support


### Ubuntu Linux with SDL2
* install SDL2 - sudo apt-get install libsdl2-dev
* Makefile in /src/sdl/
* Create build directory - make setup
* Run program - make run


### Jetson Nano with SDL2
* install SDL2 - sudo apt-get install libsdl2-dev
* SDL bug with ARM processors - sudo killall ibus-daemon
* Makefile in /src/cuda_sdl/
* Create build directory - make setup
* Run program - make run