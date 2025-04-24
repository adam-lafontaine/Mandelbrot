#pragma once

#include "../util/memory_buffer.hpp"
#include "../util/stack_buffer.hpp"

#define SPAN_TRANSFORM
#define SPAN_STRING


template <typename T>
class SpanView
{
public:
	T* data = 0;
	u32 length = 0;
};


using ByteView = SpanView<u8>;


/* make_view */

namespace span
{
    namespace mb = memory_buffer;
    namespace sb = stack_buffer;


    template <typename T>
    inline SpanView<T> make_view(T* data, u32 len)
    {
        SpanView<T> span{};
        span.data = data;
        span.length = len;

        return span;
    }


    template <typename T>
    inline SpanView<T> make_view(MemoryBuffer<T> const& buffer)
    {
        return make_view(buffer.data_, buffer.capacity_);
    }


    template <typename T>
    inline SpanView<T> push_span(MemoryBuffer<T>& buffer, u32 length)
    {
        SpanView<T> view{};

        auto data = mb::push_elements(buffer, length);
        
        return make_view(data, length);
    }


    template <typename T, u32 N>
    inline SpanView<T> push_span(StackBuffer<T, N>& buffer, u32 length)
    {
        SpanView<T> view{};

        auto data = sb::push_elements(buffer, length);
        if (data)
        {
            view.data = data;
            view.length = length;
        }

        return view;
    }
}


/* copy/fill */

namespace span
{
    void copy_u8(u8* src, u8* dst, u64 len);

    void copy_u8(u8* src, u8* dst1, u8* dst2, u64 len_u8);

    void fill_u8(u8* dst, u8 value, u64 len);

    void fill_u32(u32* dst, u32 value, u64 len);


    template <typename T>
    inline void copy(SpanView<T> const& src, SpanView<T> const& dst)
    {
        copy_u8((u8*)src.data, (u8*)dst.data, src.length * sizeof(T));
    }


    template <typename T>
    inline void copy(SpanView<T> const& src, SpanView<T> const& dst1, SpanView<T> const& dst2)
    {
        copy_u8((u8*)src.data, (u8*)dst1.data, (u8*)dst2.data, src.length * sizeof(T));
    }


    template <typename T>
    inline void fill_32(SpanView<T> const& dst, T value)
    {
        //static_assert(sizeof(T) == sizeof(u32));
        auto val = *((u32*)&value);
        fill_u32((u32*)dst.data, val, dst.length);
    }


    template <typename T>
    inline void fill_8(SpanView<T> const& dst, T value)
    {
        //static_assert(sizeof(T) == sizeof(u8));

        auto val = *((u8*)&value);
        fill_u8((u8*)dst.data, val, dst.length);
    }
    
    
    template <typename T>
	inline void fill(SpanView<T> const& dst, T value)
	{
        switch (sizeof(T))
        {
        case 1:
            fill_8(dst, value);
            return;
        
        case 4:
            fill_32(dst, value);
            return;
        }

        T* d = dst.data;
		for (u32 i = 0; i < dst.length; ++i)
		{
			d[i] = value;
		}
	}
}


/* ops */

namespace span
{    
    inline void add(SpanView<f32> const& a, SpanView<f32> const& b, SpanView<f32> const& dst)
    {
        constexpr u32 N = 8;
        auto len = (a.length / N) * N;

        u32 i = 0;
        for (; i < len; i += N)
        {
            dst.data[i] = a.data[i] + b.data[i];
            dst.data[i + 1] = a.data[i + 1] + b.data[i + 1];
            dst.data[i + 2] = a.data[i + 2] + b.data[i + 2];
            dst.data[i + 3] = a.data[i + 3] + b.data[i + 3];
            dst.data[i + 4] = a.data[i + 4] + b.data[i + 4];
            dst.data[i + 5] = a.data[i + 5] + b.data[i + 5];
            dst.data[i + 6] = a.data[i + 6] + b.data[i + 6];
            dst.data[i + 7] = a.data[i + 7] + b.data[i + 7];
        }

        len = a.length;
        for (; i < len; i++)
        {
            dst.data[i] = a.data[i] + b.data[i];
        }
    }
    

    inline void sub(SpanView<f32> const& a, SpanView<f32> const& b, SpanView<f32> const& dst)
    {
        constexpr u32 N = 8;
        auto len = (a.length / N) * N;

        u32 i = 0;
        for (; i < len; i += N)
        {
            dst.data[i] = a.data[i] + b.data[i];
            dst.data[i + 1] = a.data[i + 1] - b.data[i + 1];
            dst.data[i + 2] = a.data[i + 2] - b.data[i + 2];
            dst.data[i + 3] = a.data[i + 3] - b.data[i + 3];
            dst.data[i + 4] = a.data[i + 4] - b.data[i + 4];
            dst.data[i + 5] = a.data[i + 5] - b.data[i + 5];
            dst.data[i + 6] = a.data[i + 6] - b.data[i + 6];
            dst.data[i + 7] = a.data[i + 7] - b.data[i + 7];
        }

        len = a.length;
        for (; i < len; i++)
        {
            dst.data[i] = a.data[i] - b.data[i];
        }
    }


    inline void mul(SpanView<f32> const& a, f32 scalar, SpanView<f32> const& dst)
    {
        constexpr u32 N = 8;
        auto len = (a.length / N) * N;

        u32 i = 0;
        for (; i < len; i += N)
        {
            dst.data[i] = a.data[i] * scalar;
            dst.data[i + 1] = a.data[i + 1] * scalar;
            dst.data[i + 2] = a.data[i + 2] * scalar;
            dst.data[i + 3] = a.data[i + 3] * scalar;
            dst.data[i + 4] = a.data[i + 4] * scalar;
            dst.data[i + 5] = a.data[i + 5] * scalar;
            dst.data[i + 6] = a.data[i + 6] * scalar;
            dst.data[i + 7] = a.data[i + 7] * scalar;
        }

        len = a.length;
        for (; i < len; i++)
        {
            dst.data[i] = a.data[i] * scalar;
        }
    }
    

    inline f32 dot(SpanView<f32> const& a, SpanView<f32> const& b)
    {
        constexpr u32 N = 8;
        auto len = (a.length / N) * N;

        f32 sum[8] = { 0 };

        u32 i = 0;
        for (; i < len; i += N)
        {
            sum[0] += a.data[i] * b.data[i];
            sum[1] += a.data[i + 1] * b.data[i + 1];
            sum[2] += a.data[i + 2] * b.data[i + 2];
            sum[3] += a.data[i + 3] * b.data[i + 3];
            sum[4] += a.data[i + 4] * b.data[i + 4];
            sum[5] += a.data[i + 5] * b.data[i + 5];
            sum[6] += a.data[i + 6] * b.data[i + 6];
            sum[7] += a.data[i + 7] * b.data[i + 7];
        }

        f32 res = 0.0f;
        for (u32 s = 0; s < N; s++)
        {
            res += sum[s];
        }

        len = a.length;
        for (; i < len; i++)
        {
            res += a.data[i] * b.data[i];
        }

        return res;
    }
}


#ifdef SPAN_TRANSFORM

#include <functional>

template <class F>
using fn = std::function<F>;


/* transform */

namespace span
{
    template <typename S, typename D, class FUNC>
    inline void transform(SpanView<S> const& src, SpanView<D> const& dst, FUNC const& func)
    {
        constexpr u32 N = 8;
        auto len = (src.length / N) * N;

        auto s = src.data;
        auto d = dst.data;

        u32 i = 0;
        for (; i < len; i += N)
        {
            d[i] = func(s[i]);
            d[i + 1] = func(s[i + 1]);
            d[i + 2] = func(s[i + 2]);
            d[i + 3] = func(s[i + 3]);
            d[i + 4] = func(s[i + 4]);
            d[i + 5] = func(s[i + 5]);
            d[i + 6] = func(s[i + 6]);
            d[i + 7] = func(s[i + 7]);
        }

        len = src.length;
        for (; i < len; i++)
        {
            d[i] = func(s[i]);
        }
    }


    template <typename S1, typename S2, typename D, class FUNC>
    inline void transform(SpanView<S1> const& src1, SpanView<S2> const& src2, SpanView<D> const& dst, FUNC const& func)
    {
        constexpr u32 N = 8;
        auto len = (src1.length / N) * N;

        auto s1 = src1.data;
        auto s2 = src2.data;
        auto d = dst.data;

        u32 i = 0;
        for (; i < len; i += N)
        {
            d[i] = func(s1[i], s2[i]);
            d[i + 1] = func(s1[i + 1], s2[i + 1]);
            d[i + 2] = func(s1[i + 2], s2[i + 2]);
            d[i + 3] = func(s1[i + 3], s2[i + 3]);
            d[i + 4] = func(s1[i + 4], s2[i + 4]);
            d[i + 5] = func(s1[i + 5], s2[i + 5]);
            d[i + 6] = func(s1[i + 6], s2[i + 6]);
            d[i + 7] = func(s1[i + 7], s2[i + 7]);
        }

        len = src1.length;
        for (; i < len; i++)
        {
            d[i] = func(s1[i], s2[i]);
        }
    }
}

#endif // SPAN_TRANSFORM


#ifdef SPAN_STRING

#include "../stb_libs/qsprintf.hpp"


class StringView
{
public:
    char* data = 0;
    u32 capacity = 0;
    u32 length = 0;
};


/* string view */

namespace span
{
    inline constexpr u32 strlen(cstr text)
    {
        u32 len = 0;

        for (; text[len]; len++) {}

        return len;
    }


    inline constexpr int strcmp(cstr a, cstr b)
    {
        while (*a && (*a == *b))
        {
            ++a;
            ++b;
        }

        return (int)*a - (int)*b;
    }


    inline cstr to_cstr(StringView const& view)
    {
        return (cstr)view.data;
    }


    inline constexpr StringView to_string_view(cstr text)
    {
        StringView view{};

        view.data = (char*)text;
        view.capacity = strlen(text);
        view.length = view.capacity;

        return view;
    }


    inline void zero_string(StringView& view)
    {
        view.length = 0;

        fill_u8((u8*)view.data, 0, view.capacity);
    }


    inline StringView make_view(u32 capacity, MemoryBuffer<u8>& buffer)
    {
        StringView view{};

        auto data = mb::push_elements(buffer, capacity);
        if (data)
        {
            view.data = (char*)data;
            view.capacity = capacity;
            view.length = 0;
            
            zero_string(view);
        }

        return view;
    }


    inline StringView make_view(u32 capacity, char* buffer)
    {
        StringView view{};

        view.data = buffer;
        view.capacity = capacity;
        view.length = 0;

        return view;
    }


    inline void set_length(StringView& view)
    {
        view.length = view.capacity;

        for (u32 i = 0; i < view.capacity; i++)
        {
            if (!view.data[i])
            {
                view.length = i;
                return;
            }
        }
    }


    template <typename... VA_ARGS>
    inline void sprintf(StringView& view, cstr fmt, VA_ARGS... va_args)
    {
        view.length = (u32)stb::qsnprintf(view.data, (int)view.capacity, fmt, va_args...);
    }

}

#endif // SPAN_STRING


namespace span
{
    template <typename T>
    using SV = SpanView<T>;

    using SVf64 = SpanView<f64>;
    using SVf32 = SpanView<f32>;


    void fma(SVf64 const& src_a, SVf64 const& src_b, SVf64 const& src_c, SVf64 const& dst);

    void fmaf(SVf32 const& src_a, SVf32 const& src_b, SVf32 const& src_c, SVf32 const& dst);

    void clamp(SVf32 const& values, f32 min, f32 max, SVf32 const& dst);

    void sign(SVf32 const& values, SVf32 const& dst);

    void round_to_unsigned(SVf32 const& values, SV<u8> const& dst);

    void round_to_unsigned(SVf32 const& values, SV<u16> const& dst);

    void round_to_unsigned(SVf32 const& values, SV<u32> const& dst);

    void round_to_unsigned(SVf32 const& values, SV<u64> const& dst);

    void round_to_signed(SVf32 const& values, SV<i32> const& dst);

    void abs(SVf32 const& values, SVf32 const& dst);

    void abs(SV<i32> const& values, SV<i32> const& dst);

    void min(SVf32 const& src_a, SVf32 const& src_b, SVf32 const& dst);

    void min(SV<i32> const& src_a, SV<i32> const& src_b, SV<i32> const& dst);

    void min(SV<u32> const& src_a, SV<u32> const& src_b, SV<u32> const& dst);

    void max(SVf32 const& src_a, SVf32 const& src_b, SVf32 const& dst);

    void max(SV<i32> const& src_a, SV<i32> const& src_b, SV<i32> const& dst);

    void max(SV<u32> const& src_a, SV<u32> const& src_b, SV<u32> const& dst);

    void floor(SVf32 const& values, SVf32 const& dst);
}