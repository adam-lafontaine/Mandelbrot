EPP := em++
#SDL2 := `sdl2-config --cflags --libs`

# sudo killall ibus-daemon
# apt-get install libsdl2-dev libsdl2-dbg
# python3 -m http.server

build := ../../build/sdl_wasm
code  := ..
app   := $(code)/app
input := $(code)/input
utils := $(code)/utils
sdl   := $(code)/sdl
sdl_wasm := $(code)/sdl_wasm

exe_name := mandelbrot_wasm


defines_h := $(utils)/defines.hpp

index_range_h := $(utils)/index_range.hpp

stopwatch_h := $(utils)/stopwatch.hpp

types_h := $(utils)/types.hpp
types_h += $(defines_h)

input_h := $(input)/input.hpp
input_h += $(types_h)

input_state_h := $(input)/input_state.hpp
input_state += $(input_h)

sdl_input_h := $(sdl)/sdl_input.hpp
sdl_input_h += $(input_h)

app_input_h := $(app)/app_input.hpp
app_input_h += $(types_h)
app_input_h += $(input_h)

colors_h := $(app)/colors.hpp
colors_h += $(types_h)

render_include_h := $(app)/render_include.hpp
render_include_h += $(types_h)

range_list_h := $(app)/range_list.hpp
range_list_h += $(types_h)

app_h := $(app)/app.hpp
app_h += $(input_h)

app_types_h := $(app)/app_types.hpp
app_types_h += $(app_input_h)

render_h := $(app)/render.hpp
render_h += $(app_types_h)


main_c         := $(sdl_wasm)/sdl_wasm_main.cpp
main_o         := $(build)/main.o
object_files   := $(main_o)

sdl_input_c := $(sdl)/sdl_input.cpp
sdl_input_o := $(build)/sdl_input.o
object_files += $(sdl_input_o)

app_input_c := $(app)/app_input.cpp
app_input_o := $(build)/app_input.o
object_files += $(app_input_o)

app_c := $(app)/app.cpp
app_o := $(build)/app.o
object_files += $(app_o)

render_c := $(app)/render.cpp
render_o := $(build)/render.o
object_files += $(render_o)



program_exe := $(build)/$(exe_name)

#EPP += -O3 -DNDEBUG

$(render_o): $(render_c) $(render_h) $(render_include_h) $(colors_h) $(index_range_h) $(range_list_h)
	@echo "\n $(render_o)"
	$(EPP) -o $@ -c $<

$(app_o): $(app_c) $(app_h) $(render_h) $(app_input_h)
	@echo "\n $(app_o)"
	$(EPP) -o $@ -c $<

$(app_input_o): $(app_input_c) $(app_input_h) $(app_h) $(render_include_h)
	@echo "\n $(app_input_o)"
	$(EPP) -o $@ -c $<

$(sdl_input_o): $(sdl_input_c) $(sdl_input_h) $(input_state_h)
	@echo "\n $(sdl_input_o)"
	$(EPP) -s USE_SDL=2 -o $@ -c $<

$(main_o): $(main_c) $(app_h) $(stopwatch_h) $(sdl_input_h)
	@echo "\n $(main_o)"
	$(EPP) -s USE_SDL=2 -o $@ -c $<

$(program_exe): $(object_files)
	@echo "\n $(exe_name)"
	$(EPP) -o $@ $+

build: $(program_exe)

run: build
	$(program_exe)
	@echo "\n"

clean:
	rm -rfv $(build)/*

setup:
	mkdir -p $(build)