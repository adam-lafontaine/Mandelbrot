#pragma once

#include <random>


namespace rng
{
    class iUniform
    {
    private:
        std::random_device rd;
        std::default_random_engine eng;
        std::uniform_int_distribution<int> dist;

    public:

        iUniform(int min, int max)
        {
            eng = std::default_random_engine(rd());

            if (min < max)
            {
                dist = std::uniform_int_distribution<int>(min, max);
            }
            else
            {
                dist = std::uniform_int_distribution<int>(max, min);
            }            
        }


        int get() { return dist(eng); }
    };
}