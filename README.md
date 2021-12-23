Mandelbrot Renderer
* Visual Studio solution to run on Windows
* Ubuntu Linux with SDL2
 * Makefile in /src/sdl/
 * install sdl - apt-get install libsdl2-dev libsdl2-dbg
* Jetson Nano with SDL2
 * Makefile in /src/cuda_sdl/
 * install sdl - apt-get install libsdl2-dev libsdl2-dbg
 * SDL bug with ARM processors - sudo killall ibus-daemon


Features
* Explore the mandelbrot set using keyboard controls
* Controller input available for linux
* Select color scheme
* Color contrast maximized for the current visible section


Keyboard Controls
* Zoom in with '+' (numpad)
* Zoom out with '-' (numpad)
* Pan left, right, up, down with arrow keys
* Increase zoom rate with '*'
* Decrease zoom rate with '/'
* Increase resolution with 'F'
* Decrease resolution with 'D'
* Change color scheme with '1' - '6'
* End program with ESC key


Controller Controls (Linux only)
* Zoom in and out with left thumbstick
* Pan left, right, up, down with right thumbstick
* Increase zoom rate with right trigger
* Decrease zoom rate with left trigger
* Increase resolution with right bumper
* Decrease resolution with left bumber
* Cycle color scheme with D pad
* End program with B button