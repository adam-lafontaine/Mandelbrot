GPP := g++-11 -std=c++20 -mavx -mavx2 -mfma

#GPP += -DNDEBUG -O3

GPP += -DIMAGE_READ
GPP += -DIMAGE_WRITE
GPP += -DALLOC_COUNT

NO_FLAGS := 
SDL2 := `sdl2-config --cflags --libs`
SDL_AUDIO := -lSDL2_mixer
OPENGL := -lGL -ldl
THREAD := -lpthread -ltbb

ALL_LFLAGS := $(SDL2) $(SDL_AUDIO) $(OPENGL) $(THREAD)

root       := ../../../..

engine     := $(root)/engine
build      := $(engine)/build/sdl_ogl
src        := $(engine)/src

pltfm      := $(src)/pltfm_sdl_ogl/ubuntu
pltfm_imgui := $(src)/pltfm_sdl_ogl/imgui_sdl2_ogl3

libs   := $(root)/libs

exe := exe_sdl_ogl

program_exe := $(build)/$(exe)


#*** imgui ***

imgui := $(libs)/imgui

imgui_options_h := $(pltfm_imgui)/imgui_options.hpp

imgui_h := $(imgui)/imgui.h

imgui_include_h := $(pltfm_imgui)/imgui_include.hpp
imgui_include_h += $(imgui_options_h)
imgui_include_h += $(imgui_h)
imgui_include_h += $(imgui)/backends/imgui_impl_sdl2.h
imgui_include_h += $(imgui)/backends/imgui_impl_opengl3.h

#**************


#*** libs/util ***

util := $(libs)/util

types_h := $(util)/types.hpp

numeric_h := $(util)/numeric.hpp
numeric_h += $(types_h)

stopwatch_h := $(util)/stopwatch.hpp
rng_h := $(util)/rng.hpp

#************


#*** alloc_type ***

alloc_type := $(libs)/alloc_type

alloc_type_h := $(alloc_type)/alloc_type.hpp
alloc_type_h += $(types_h)

alloc_type_c := $(alloc_type)/alloc_type.cpp
alloc_type_c += $(alloc_type_h)

#*************


#*** memory_buffer ***

memory_buffer_h := $(util)/memory_buffer.hpp
memory_buffer_h += $(alloc_type_h)

#***********


#*** stack_buffer ***

stack_buffer_h := $(util)/stack_buffer.hpp
stack_buffer_h += $(types_h)

#***********


#*** stb_libs ***

stb_libs := $(libs)/stb_libs

qsprintf_h := $(stb_libs)/qsprintf.hpp
stb_image_options_h := $(stb_libs)/stb_image_options.hpp

stb_libs_c := $(stb_libs)/stb_libs.cpp
stb_libs_c += $(stb_image_options_h)

#*************


#*** span ***

span := $(libs)/span

span_h := $(span)/span.hpp
span_h += $(memory_buffer_h)
span_h += $(stack_buffer_h)
span_h += $(qsprintf_h)

span_c := $(span)/span.cpp

#************


#*** image ***

image := $(libs)/image

image_h := $(image)/image.hpp
image_h += $(span_h)

image_c := $(image)/image.cpp
image_c += $(image_h)
image_c += $(numeric_h)
image_c += $(stb_image_options_h)

#*************


#*** ascii_image ***

ascii_image := $(libs)/ascii_image

ascii_image_h := $(ascii_image)/ascii_image.hpp
ascii_image_h += $(image_h)
ascii_image_h += $(span_h)
ascii_image_h += $(numeric_h)

ascii_image_c := $(ascii_image)/ascii_image.cpp
ascii_image_c += $(ascii_image_h)
ascii_image_c += $(ascii_image)/ascii_5.cpp
ascii_image_c += $(ascii_image)/ascii_joystick_8.cpp

#************


#*** input ***

input := $(libs)/input

input_h := $(input)/input.hpp
input_h += $(types_h)
input_h += $(input)/keyboard_input.hpp
input_h += $(input)/mouse_input.hpp
input_h += $(input)/controller_input.hpp
input_h += $(input)/joystick_input.hpp

input_state_h := $(input)/input_state.hpp
input_state_h += $(input_h)

#*************


#*** output ***

output := $(libs)/output

window_h := $(output)/window.hpp
audio_h := $(output)/audio.hpp

#*************


#*** sdl2 ***

sdl2 := $(libs)/sdl2

sdl_include_h := $(sdl2)/sdl_include.hpp

sdl_input_c := $(sdl2)/sdl_input.cpp
sdl_input_c += $(input_state_h)
sdl_input_c += $(numeric_h)
sdl_input_c += $(sdl_include_h)

sdl_window_c := $(sdl2)/sdl_window.cpp
sdl_window_c += $(window_h)
sdl_window_c += $(alloc_type_h)
sdl_window_c += $(sdl_include_h)

sdl_audio_c := $(sdl2)/sdl_audio.cpp

#************


#*** io_test ********

io_test := $(src)/io_test

io_test_app := $(io_test)/app
io_test_res := $(io_test)/res

io_test_h := $(io_test_app)/app.hpp

io_test_c := $(io_test_app)/app.cpp
io_test_c += $(io_test_h)
io_test_c += $(numeric_h)
io_test_c += $(ascii_image_h)

# assets.cpp
io_test_c += $(io_test_app)/assets.cpp
io_test_c += $(audio_h)
io_test_c += $(io_test_res)/asset_sizes.cpp

#************


#*** game_mbt ***

game_mbt := $(root)/game_mbt
game_mbt_src := $(game_mbt)/src
game_mbt_app := $(game_mbt_src)/app

game_mbt_h := $(game_mbt_app)/app.hpp
game_mbt_h += $(input_h)
game_mbt_h += $(image_h)

game_mbt_libs := $(game_mbt_app)/app_libs.cpp
game_mbt_libs += $(alloc_type_c)
game_mbt_libs += $(span_c)
game_mbt_libs += $(image_c)
game_mbt_libs += $(stb_libs_c)
game_mbt_libs += $(ascii_image_c)
game_mbt_libs += $(sdl_input_c)
game_mbt_libs += $(sdl_window_c)
game_mbt_libs += $(sdl_audio_c)

game_mbt_c := $(game_mbt_app)/app.cpp
game_mbt_c += $(game_mbt_h)
game_mbt_c += $(alloc_type_h)
game_mbt_c += $(numeric_h)
game_mbt_c += $(stopwatch_h)
game_mbt_c += $(rng_h)
game_mbt_c += $(game_mbt_libs)
game_mbt_c += $(game_mbt_app)/colors.cpp
game_mbt_c += $(game_mbt_app)/map_input.cpp

#*************


#*** game state ***

game_state := $(src)/game_state

game_state_h := $(game_state)/game_state.hpp
game_state_h += $(input_h)
game_state_h += $(image_h)

game_state_c := $(game_state)/game_state.cpp
game_state_c += $(game_state_h)
game_state_c += $(game_mbt_c)

#***************


#*** ui ***

ui := $(src)/ui

ui_h := $(ui)/ui.hpp
ui_h += $(imgui_h)
ui_h += $(stopwatch_h)
ui_h += $(qsprintf_h)
ui_h += $(alloc_type_h)

ui_h += $(ui)/diagnostics_window.hpp
ui_h += $(game_state_h)

ui_h += $(ui)/input_frames_window.hpp

#*************


#*** input_display ***

input_display := $(src)/input_display

input_display_h := $(input_display)/input_display.hpp
input_display_h += $(input_h)
input_display_h += $(image_h)

#***************


#*** main cpp ***

main_c := $(pltfm)/engine_main_ubuntu.cpp
main_o := $(build)/main.o
obj    := $(main_o)

main_dep := $(imgui_include_h)
main_dep += $(ui_h)
main_dep += $(io_test_h)

main_dep += $(pltfm)/main_o.cpp
main_dep += $(game_state_c)
main_dep += $(io_test_c)

#************


#*** imgui cpp ***

imgui_c := $(pltfm_imgui)/imgui_o.cpp
imgui_o := $(build)/imgui.o
obj     += $(imgui_o)

imgui_dep := $(imgui_options_h)

#****************


#*** engine ***


$(main_o): $(main_c) $(main_dep)
	@echo "\n  main"
	$(GPP) -o $@ -c $< $(SDL2) $(OPENGL)


$(imgui_o): $(imgui_c) $(imgui_dep)
	@echo "\n  imgui"
	$(GPP) -o $@ -c $< $(SDL2) $(OPENGL)

#**************


$(program_exe): $(obj)
	@echo "\n  program_exe"
	$(GPP) -o $@ $+ $(ALL_LFLAGS)


build: $(program_exe)


run: build
	$(program_exe)
	@echo "\n"


clean:
	rm -rfv $(build)/*

setup:
	mkdir -p $(build)