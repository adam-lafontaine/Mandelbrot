EPP := em++

# apt-get install libsdl2-dev
# python3 -m http.server 8080

build := ../../build/sdl_wasm
web := $(build)/web
code  := ..
app   := $(code)/app
input := $(code)/input
utils := $(code)/utils
sdl   := $(code)/sdl
sdl_wasm := $(code)/sdl_wasm

exe_name := mandelbrot_wasm
html_shell := ./out_template.html

EPP_FLAGS := -s USE_SDL=2
EPP_FLAGS += -sALLOW_MEMORY_GROWTH
EPP_FLAGS += --shell-file $(html_shell)


defines_h := $(utils)/defines.hpp

index_range_h := $(utils)/index_range.hpp

types_h := $(utils)/types.hpp
types_h += $(defines_h)

input_h := $(input)/input.hpp
input_h += $(types_h)

input_state_h := $(input)/input_state.hpp
input_state += $(input_h)

sdl_include_h := $(sdl)/sdl_include.hpp
sdl_include_h += $(input_h)

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



program_exe := $(build)/$(exe_name).html

EPP += -O2

$(render_o): $(render_c) $(render_h) $(render_include_h) $(colors_h) $(index_range_h) $(range_list_h)
	@echo "\n $(render_o)"
	$(EPP) -o $@ -c $<

$(app_o): $(app_c) $(app_h) $(render_h) $(app_input_h)
	@echo "\n $(app_o)"
	$(EPP) -o $@ -c $<

$(app_input_o): $(app_input_c) $(app_input_h) $(app_h) $(render_include_h)
	@echo "\n $(app_input_o)"
	$(EPP) -o $@ -c $<

$(sdl_input_o): $(sdl_input_c) $(sdl_include_h) $(input_state_h)
	@echo "\n $(sdl_input_o)"
	$(EPP) -o $@ -c $<

$(main_o): $(main_c) $(app_h) $(sdl_include_h)
	@echo "\n $(main_o)"
	$(EPP) -s -o $@ -c $<

$(program_exe): $(object_files)
	@echo "\n $(exe_name)"
	$(EPP) $(EPP_FLAGS) -o $@ $+
	

build: $(program_exe)


html:
	@echo "\n $(html_shell)"
	$(EPP) $(EPP_FLAGS) -o $(program_exe) $(object_files)


clean:
	rm -rfv $(build)/*

setup:
	mkdir -p $(build)
	mkdir -p $(web)