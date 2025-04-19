#include "span.hpp"
#include "../util/numeric.hpp"


/* fma */

namespace span
{
    namespace num = numeric;

    


    void fma(SVf64 const& src_a, SVf64 const& src_b, SVf64 const& src_c, SVf64 const& dst)
    {
        auto len = src_a.length;

        auto a = src_a.data;
        auto b = src_b.data;
        auto c = src_c.data;
        auto d = dst.data;

        assert(src_b.length == len);
        assert(src_c.length == len);
        assert(dst.length == len);

        for (u32 i = 0; i < len; i++)
        {
            d[i] = a[i] * b[i] + c[i];
        }
    }


    void fmaf(SVf32 const& src_a, SVf32 const& src_b, SVf32 const& src_c, SVf32 const& dst)
    {
        auto len = src_a.length;

        auto a = src_a.data;
        auto b = src_b.data;
        auto c = src_c.data;
        auto d = dst.data;

        assert(src_b.length == len);
        assert(src_c.length == len);
        assert(dst.length == len);

        for (u32 i = 0; i < len; i++)
        {
            d[i] = a[i] * b[i] + c[i];
        }
    }
}


/* clamp */

namespace span
{
    template <typename T>
    static void t_clamp(SV<T> const& values, T min, T max, SV<T> const& dst)
    {
        auto len = values.length;

        auto v = values.data;
        auto d = dst.data;

        for (u32 i = 0; i < len; i++)
        {
            d[i] = num::clamp(v[i], min, max);
        }
    }


    void clamp(SVf32 const& values, f32 min, f32 max, SVf32 const& dst)
    {
        assert(values.length == dst.length);

        t_clamp(values, min, max, dst);
    }
}


/* sign */

namespace span
{
    void sign(SVf32 const& values, SVf32 const& dst)
    {
        assert(values.length == dst.length);

        auto len = values.length;

        auto v = values.data;
        auto d = dst.data;

        for (u32 i = 0; i < len; i++)
        {
            d[i] = num::sign_f32(v[i]);
        }
    }
}


/* round */

namespace span
{
    template <typename T>
    static inline void t_round_to_unsigned(SVf32 const& values, SV<T> const& dst)
    {
        static_assert(num::is_unsigned<T>());

        auto len = values.length;

        auto v = values.data;
        auto d = dst.data;

        for (u32 i = 0; i < len; i++)
        {
            d[i] = num::round_to_unsigned<T>(v[i]);
        }
    }


    template <typename T>
    static inline void t_round_to_signed(SVf32 const& values, SV<T> const& dst)
    {
        static_assert(!num::is_unsigned<T>());

        constexpr f32 P = 1.0f;
        constexpr f32 N = -1.0f;
        constexpr f32 Z = 0.0f;

        auto len = values.length;

        auto v = values.data;
        auto d = dst.data;

        for (u32 i = 0; i < len; i++)
        {
            d[i] = num::round_to_signed<T>(v[i]);
        }
    }


    void round_to_unsigned(SVf32 const& values, SV<u8> const& dst)
    {
        assert(values.length == dst.length);

        t_round_to_unsigned(values, dst);
    }


    void round_to_unsigned(SVf32 const& values, SV<u16> const& dst)
    {
        assert(values.length == dst.length);

        t_round_to_unsigned(values, dst);
    }


    void round_to_unsigned(SVf32 const& values, SV<u32> const& dst)
    {
        assert(values.length == dst.length);

        t_round_to_unsigned(values, dst);
    }


    void round_to_unsigned(SVf32 const& values, SV<u64> const& dst)
    {
        assert(values.length == dst.length);

        t_round_to_unsigned(values, dst);
    }


    void round_to_signed(SVf32 const& values, SV<i32> const& dst)
    {
        assert(values.length == dst.length);

        t_round_to_signed(values, dst);
    }
}


/* abs */

namespace span
{
    template <typename T>
    static inline void t_abs(SV<T> const& values, SV<T> const& dst)
    {
        auto len = values.length;

        auto v = values.data;
        auto d = dst.data;

        for (u32 i = 0; i < len; i++)
        {
            d[i] = num::abs(v[i]);
        }
    }


    void abs(SVf32 const& values, SVf32 const& dst)
    {
        assert(values.length == dst.length);

        t_abs<f32>(values, dst);
    }


    void abs(SV<i32> const& values, SV<i32> const& dst)
    {
        assert(values.length == dst.length); 

        t_abs<i32>(values, dst);
    }
}


/* min */

namespace span
{
    template <typename T>
    static inline void t_min(SV<T> const& src_a, SV<T> const& src_b, SV<T> const& dst)
    {
        auto len = src_a.length;

        auto a = src_a.data;
        auto b = src_b.data;
        auto d = dst.data;

        for (u32 i = 0; i < len; i++)
        {
            d[i] = num::min(a[i], b[i]);
        }
    }


    void min(SVf32 const& src_a, SVf32 const& src_b, SVf32 const& dst)
    {
        assert(src_a.length == dst.length);
        assert(src_b.length == dst.length);

        t_min(src_a, src_b, dst);
    }


    void min(SV<i32> const& src_a, SV<i32> const& src_b, SV<i32> const& dst)
    {
        assert(src_a.length == dst.length);
        assert(src_b.length == dst.length);

        t_min(src_a, src_b, dst);
    }


    void min(SV<u32> const& src_a, SV<u32> const& src_b, SV<u32> const& dst)
    {
        assert(src_a.length == dst.length);
        assert(src_b.length == dst.length);

        t_min(src_a, src_b, dst);
    }
}


/* max */

namespace span
{
    template <typename T>
    static inline void t_max(SV<T> const& src_a, SV<T> const& src_b, SV<T> const& dst)
    {
        auto len = src_a.length;

        auto a = src_a.data;
        auto b = src_b.data;
        auto d = dst.data;

        for (u32 i = 0; i < len; i++)
        {
            d[i] = num::max(a[i], b[i]);
        }
    }


    void max(SVf32 const& src_a, SVf32 const& src_b, SVf32 const& dst)
    {
        assert(src_a.length == dst.length);
        assert(src_b.length == dst.length);

        t_max(src_a, src_b, dst);
    }


    void max(SV<i32> const& src_a, SV<i32> const& src_b, SV<i32> const& dst)
    {
        assert(src_a.length == dst.length);
        assert(src_b.length == dst.length);

        t_max(src_a, src_b, dst);
    }


    void max(SV<u32> const& src_a, SV<u32> const& src_b, SV<u32> const& dst)
    {
        assert(src_a.length == dst.length);
        assert(src_b.length == dst.length);

        t_max(src_a, src_b, dst);
    }
}


/* floor */

namespace span
{
    template <typename T>
    static inline void t_floor(SV<T> const& values, SV<T> const& dst)
    {
        auto len = values.length;

        auto v = values.data;
        auto d = dst.data;

        for (u32 i = 0; i < len; i++)
        {
            d[i] = num::floor(v[i]);
        }
    }


    void floor(SVf32 const& values, SVf32 const& dst)
    {
        assert(values.length == dst.length);

        t_floor(values, dst);
    }
}