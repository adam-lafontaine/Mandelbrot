CPP := g++-11

CPPFLAGS := -std=c++20
CPPFLAGS += -mavx -mfma
#CPPFLAGS += -O3
#CPPFLAGS += -DNDEBUG

EXE := mandelbrot

# apt-get install libsdl2-dev
# apt-get install libsdl2-mixer-dev

LDFLAGS := -lSDL2
LDFLAGS += -lSDL2_mixer

ROOT := ../../..

BUILD := $(ROOT)/build/sdl

RES := $(ROOT)/src/res

#BIN_DATA := $(RES)/music/asteroids_data.bin

OUT := $(BUILD)/$(EXE)

SRC := mbt_sdl2_main.cpp


#****************

build:
	$(CPP) $(CPPFLAGS) -o $(OUT) $(SRC) $(LDFLAGS)
#	cp $(BIN_DATA) $(BUILD)


clean:
	rm -rfv $(BUILD)/*


setup:
	mkdir -p $(BUILD)


delete:
	rm -rfv $(BUILD)