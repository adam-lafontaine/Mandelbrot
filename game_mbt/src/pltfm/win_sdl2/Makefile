CPP := g++

CPPFLAGS := -std=c++20
CPPFLAGS += -mavx -mfma
CPPFLAGS += -O3
CPPFLAGS += -DNDEBUG
CPPFLAGS += -mwindows
CPPFLAGS += -DNO_AUDIO

EXE := mandelbrot.exe

SDL := C:\SDL2\SDL2-2.30.5\x86_64-w64-mingw32
#SDL_MIXER := C:\SDL2\SDL2_mixer-2.8.0\x86_64-w64-mingw32

SDL_DLL := $(SDL)/bin/SDL2.dll
#SDL_MIXER_DLL := $(SDL_MIXER)/bin/SDL2_mixer.dll

ISDL := -I$(SDL)\include -I$(SDL)\include\SDL2
#ISDL_MIXER=-I$(SDL_MIXER)\include

IFLAGS := $(ISDL)
#IFLAGS += $(ISDL_MIXER)

LDSDL := -L$(SDL)\lib
#LDSDL_MIXER := -L$(SDL_MIXER)\lib

LDFLAGS := $(LDSDL) -lSDL2 
#LDFLAGS += $(LDSDL_MIXER) -lSDL2_mixer

ROOT := ../../..

BUILD := $(ROOT)/build/win_sdl
O_BUILD := $(BUILD)/o

#RES := $(ROOT)/src/res
#BIN_DATA := $(RES)/music/asteroids_data.bin

OUT := $(BUILD)/$(EXE)

SRC := mbt_win_sdl2_main.cpp

#************

program_exe := $(OUT)


#*** main cpp ***

main_c := $(SRC)
main_o := $(O_BUILD)/main.o

obj := $(main_o)

main_dep := 

#***************


#*** icon ******

icon_rc := mandelbrot.rc
icon_o := $(O_BUILD)/icon.o

obj += $(icon_o)

icon_dep := icon_mandelbrot.ico

#****************


$(main_o): $(main_c) $(main_dep)
	@echo "  --- main ---"
	$(CPP) $(CPPFLAGS) $(IFLAGS) -o $@ -c $< $(LDFLAGS)


$(icon_o): $(icon_rc) $(icon_dep)
	@echo "  --- icon ---"
	@windres -i $< -o $@


$(program_exe): $(obj)
	@echo "  --- link ---"
	$(CPP) $(CPPFLAGS) -o $@ $+ $(LDFLAGS)
	cp $(SDL_DLL) $(BUILD)
#	cp $(SDL_MIXER_DLL) $(BUILD)
#	cp $(BIN_DATA) $(BUILD)
#	rm $(main_o)


build: $(program_exe)


clean:
	rm -fv $(O_BUILD)/*
	rm -fv $(BUILD)/*


setup:
	mkdir -p $(O_BUILD)


delete:
	rm -rfv $(BUILD)