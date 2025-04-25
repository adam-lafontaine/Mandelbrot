
#include "../../../libs/alloc_type/alloc_type.cpp"
#include "../../../libs/span/span.cpp"
#include "../../../libs/image/image.cpp"
#include "../../../libs/stb_libs/stb_libs.cpp"
#include "../../../libs/ascii_image/ascii_image.cpp"

#include "../../../libs/sdl2/sdl_input.cpp"
#include "../../../libs/sdl2/sdl_window.cpp"
#include "../../../libs/sdl2/sdl_audio.cpp"

// TODO: processing/cuda etc.
#include "colors.cpp"


//#define CAN_TBB

#ifdef CAN_TBB

#if __has_include(<tbb/parallel_for.h>)
#include "mbt_process_tbb.cpp"
#else
#include "mbt_process_seq.cpp"
#endif

#else
#include "mbt_process_seq.cpp"
#endif