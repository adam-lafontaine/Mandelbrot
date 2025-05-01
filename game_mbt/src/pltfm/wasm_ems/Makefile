
# python3 -m http.server 8080


EPP := em++

EPPFLAGS := -std=c++20

# EPPFLAGS += -mavx -msimd128

#EPPFLAGS += -O3
#EPPFLAGS += -DNDEBUG

#EPPFLAGS += -DWASM_PRINTF

# EPPFLAGS += -sASSERTIONS
EPPFLAGS += -s USE_SDL=2
EPPFLAGS += -DNO_AUDIO
#EPPFLAGS += -s USE_SDL_MIXER=2
#EPPFLAGS += -s USE_OGG=1
EPPFLAGS += -s ALLOW_MEMORY_GROWTH

EPPFLAGS += -DSDL2_WASM

EXE := mandelbrot

HTMLFLAGS := 
HTMLFLAGS += --shell-file out_template.html


LDFLAGS := 

ROOT := ../../..

BUILD := $(ROOT)/build/wasm_ems

OUT := $(BUILD)/$(EXE).html

OUT_WASM := $(BUILD)/$(EXE).wasm
OUT_JS   := $(BUILD)/$(EXE).js

CMS := $(ROOT)/../../CMS/sm/wasm

SRC := mandelbrot_wasm_main.cpp


#**********************

build:
	$(EPP) $(EPPFLAGS) $(HTMLFLAGS) -o $(OUT) $(SRC) $(LDFLAGS)


cms:
	cp -u -v $(OUT_WASM) $(CMS)
	cp -u -v $(OUT_JS) $(CMS)


clean:
	rm -rfv $(BUILD)/*


setup:
	mkdir -p $(BUILD)


delete:
	rm -rfv $(BUILD)