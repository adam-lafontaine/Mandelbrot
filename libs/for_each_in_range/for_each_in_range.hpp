#pragma once

#include <functional>


namespace for_each_in_range
{
    using u64_f = std::function<void(u64 id)>;
    using i64_f = std::function<void(i64 id)>;

    using uXY_f = std::function<void(u64 x, u64 y)>;
    using sXY_f = std::function<void(i64 x, i64 y)>;

    using uPt2D = Point2D<u64>;
    using sPt2D = Point2D<i64>;

    using uPt_f = std::function<void(uPt2D const&)>;
    using sPt_f = std::function<void(sPt2D const&)>;


    namespace seq
    {
        void for_each_in_range(u64 size, u64_f const& id_func);

        void for_each_in_range(u64 begin, u64 end, u64_f const& id_func);        

        void for_each_in_range(i64 begin, i64 end, i64_f const& id_func);



        void for_each_in_range_2d(u64 width, u64 height, uPt_f const& pt_func);

        void for_each_in_range_2d(uPt2D const& first, uPt2D const& last, uPt_f const& pt_func);

        void for_each_in_range_2d(sPt2D const& first, sPt2D const& last, sPt_f const& pt_func);



        void for_each_in_range_2d(u64 width, u64 height, uXY_f const& xy_func);

        void for_each_in_range_2d(u64 x_begin, u64 x_end, u64 y_begin, u64 y_end, uXY_f const& xy_func);

        void for_each_in_range_2d(i64 x_begin, i64 x_end, i64 y_begin, i64 y_end, sXY_f const& xy_func);
        
    }


    namespace par
    {
        void for_each_in_range(u64 size, u64_f const& id_func);

        void for_each_in_range(u64 begin, u64 end, u64_f const& id_func);        

        void for_each_in_range(i64 begin, i64 end, i64_f const& id_func);



        void for_each_in_range_2d(u64 width, u64 height, uPt_f const& pt_func);

        void for_each_in_range_2d(uPt2D const& first, uPt2D const& last, uPt_f const& pt_func);

        void for_each_in_range_2d(sPt2D const& first, sPt2D const& last, sPt_f const& pt_func);



        void for_each_in_range_2d(u64 width, u64 height, uXY_f const& xy_func);

        void for_each_in_range_2d(u64 x_begin, u64 x_end, u64 y_begin, u64 y_end, uXY_f const& xy_func);

        void for_each_in_range_2d(i64 x_begin, i64 x_end, i64 y_begin, i64 y_end, sXY_f const& xy_func);
    }
}
