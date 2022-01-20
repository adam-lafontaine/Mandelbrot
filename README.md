# Mandelbrot Renderer

## Features
* Explore the mandelbrot set using keyboard controls
* Controller input available for Linux
* Select color scheme
* Color contrast maximized for the current visible section


### Keyboard Controls
* Zoom in with '+' (numpad)
* Zoom out with '-' (numpad)
* Pan left, right, up, down with arrow keys
* Increase zoom rate with '*'
* Decrease zoom rate with '/'
* Increase resolution with 'F'
* Decrease resolution with 'D'
* Change color scheme with '1' - '6'
* End program with ESC key


### Controller Controls (Xinput, Linux only)
* Zoom in and out with left thumbstick
* Pan left, right, up, down with right thumbstick
* Increase zoom rate with right trigger
* Decrease zoom rate with left trigger
* Increase resolution with right bumper
* Decrease resolution with left bumper
* Cycle color scheme with D pad
* End program with B button


### Visual Studio solution to run on Windows
* Compile and run
* No third party libraries to install
* No controller support


### Ubuntu Linux with SDL2
* install sdl - sudo apt-get install libsdl2-dev
* If processor does not support std::execution - uncomment #define NO_CPP_17 in /src/utils/types.hpp
* Makefile in /src/sdl/
* Create build directory - make setup
* Run program - make run


### Jetson Nano with SDL2
* install sdl - sudo apt-get install libsdl2-dev
* SDL bug with ARM processors - sudo killall ibus-daemon
* Makefile in /src/cuda_sdl/
* Create build directory - make setup
* Run program - make run