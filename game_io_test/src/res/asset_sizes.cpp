// asset_sizes.cpp

const struct {

    struct {
        struct { unsigned size; unsigned offset; } keyboard;
        struct { unsigned size; unsigned offset; } controller;
        struct { unsigned size; unsigned offset; } mouse;
        struct { unsigned size; unsigned offset; } arrow;
    } masks;

    struct {
        struct { unsigned size; unsigned offset; } game_02;
        struct { unsigned size; unsigned offset; } game_03;
        struct { unsigned size; unsigned offset; } game_00;
        struct { unsigned size; unsigned offset; } game_01;
    } music;

    struct {
        struct { unsigned size; unsigned offset; } confirmation_002;
        struct { unsigned size; unsigned offset; } laserRetro_000;
        struct { unsigned size; unsigned offset; } open_001;
        struct { unsigned size; unsigned offset; } explosionCrunch_003;
    } sfx;

}
asset_sizes = 
{
    {
        { 2588, 0 },
        { 2094, 2588 },
        { 807, 4682 },
        { 169, 5489 },
    },
    {
        { 924117, 5658 },
        { 2214212, 929775 },
        { 2923535, 3143987 },
        { 609497, 6067522 },
    },
    {
        { 14169, 6677019 },
        { 12588, 6691188 },
        { 9834, 6703776 },
        { 54824, 6713610 },
    },
};
