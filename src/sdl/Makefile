GPP := g++-10 -std=c++17
SDL2 := `sdl2-config --cflags --libs`

# sudo killall ibus-daemon
# apt-get install libsdl2-dev libsdl2-dbg

build := ../../build/sdl
code  := ..
app   := $(code)/app
input := $(code)/input
utils := $(code)/utils
sdl   := $(code)/sdl

exe_name := mandelbrot

main_c         := $(sdl)/sdl_main.cpp
main_o         := $(build)/main.o
object_files   := $(main_o)

types_h := $(app)/types.hpp
types_h += $(utils)/types.hpp

app_h := $(app)/app.hpp

app_c := $(app)/app.cpp
app_o := $(build)/app.o
object_files += $(app_o)

controls_h := $(app)/input_controls.hpp

render_h := $(app)/render.hpp
render_h += $(app)/colors.hpp

render_c := $(app)/render.cpp
render_o := $(build)/render.o
object_files += $(render_o)

input_h := $(input)/input.hpp
input_h += $(input)/button_state.hpp
input_h += $(input)/keyboard.hpp
input_h += $(input)/mouse.hpp
input_h += $(input)/controller.hpp
input_h += $(sdl)/sdl_input.hpp

input_c := $(sdl)/sdl_input.cpp
input_o := $(build)/input.o
object_files += $(input_o)

program_exe := $(build)/$(exe_name)

GPP += -O3

$(render_o): $(render_c) $(render_h) $(types_h)
	@echo "\n $(render_o)"
	$(GPP) -o $@ -c $<

$(app_o): $(app_c) $(app_h) $(input_h) $(render_h) $(controls_h) $(types_h)
	@echo "\n $(app_o)"
	$(GPP) -o $@ -c $<

$(input_o): $(input_c) $(input_h)
	@echo "\n $(input_o)"
	$(GPP) -o $@ -c $< $(SDL2)

$(main_o): $(main_c) $(app_h) $(input_h)
	@echo "\n $(main_o)"
	$(GPP) -o $@ -c $< $(SDL2)

$(program_exe): $(object_files)
	@echo "\n $(exe_name)"
	$(GPP) -o $@ $+ $(SDL2)

build: $(program_exe)

run: build
	$(program_exe)
	@echo "\n"

clean:
	rm -rfv $(build)/*

setup:
	mkdir -p $(build)