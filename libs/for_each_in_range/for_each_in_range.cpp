#include "index_range.hpp"
#include "point_range_2d.hpp"
#include "for_each_in_range.hpp"

#include <algorithm>
#include <execution>


namespace for_each_in_range
{
    UnsignedPointRange2D to_unsigned_range_2d(uPt2D const& first, uPt2D const& last)
    {
        UnsignedPointRange2D::pt const range_first = { first.x, first.y };
        UnsignedPointRange2D::pt const range_last = { last.x, last.y };

        return UnsignedPointRange2D (range_first, range_last);
    }

    UnsignedPointRange2D to_unsigned_range_2d(uInt x_begin, uInt x_end, uInt y_begin, uInt y_end)
    {
        UnsignedPointRange2D::pt const first = { x_begin, y_begin };
        UnsignedPointRange2D::pt const last = { x_end - 1, y_end - 1 };
        
        return UnsignedPointRange2D(first, last);
    }


    SignedPointRange2D to_signed_range_2d(sPt2D const& first, sPt2D const& last)
    {
        SignedPointRange2D::pt range_first = { first.x, first.y };
        SignedPointRange2D::pt range_last = { last.x, last.y };

        return SignedPointRange2D(range_first, range_last);
    }


    SignedPointRange2D to_signed_range_2d(sInt x_begin, sInt x_end, sInt y_begin, sInt y_end)
    {
        SignedPointRange2D::pt const first = { x_begin, y_begin };
        SignedPointRange2D::pt const last = { x_end - 1, y_end - 1 };

        return SignedPointRange2D(first, last);
    }


    namespace seq
    {
        static void execute(UnsignedRange& ids, uInt_f const& id_func)
        {
            std::for_each(ids.begin(), ids.end(), id_func);
        }


        static void execute(SignedRange& ids, sInt_f const& id_func)
        {
            std::for_each(ids.begin(), ids.end(), id_func);
        }


        static void execute(UnsignedPointRange2D& pts, uPt_f const& pt_func)
        {
            auto const r_pt_func = [&](auto const& pt) { pt_func({ pt.x, pt.y }); };

            std::for_each(pts.begin(), pts.end(), r_pt_func);
        }


        static void execute(SignedPointRange2D& pts, sPt_f const& pt_func)
        {
            auto const r_pt_func = [&](auto const& pt) { pt_func({ pt.x, pt.y }); };

            std::for_each(pts.begin(), pts.end(), r_pt_func);
        }



        void for_each_in_range(uInt size, uInt_f const& id_func)
        {
            UnsignedRange ids(size);
            execute(ids, id_func);
        }
        

        void for_each_in_range(uInt begin, uInt end, uInt_f const& id_func)
        {
            UnsignedRange ids(begin, end);
            execute(ids, id_func);
        }


        void for_each_in_range(sInt begin, sInt end, sInt_f const& id_func)
        {
            SignedRange ids(begin, end);
            execute(ids, id_func);
        }


        void for_each_in_range_2d(uInt width, uInt height, uPt_f const& pt_func)
        {
            UnsignedPointRange2D pts(width, height);

            execute(pts, pt_func);
        }


        void for_each_in_range_2d(uPt2D const& first, uPt2D const& last, uPt_f const& pt_func)
        {
            auto pts = to_unsigned_range_2d(first, last);

            execute(pts, pt_func);
        }


        void for_each_in_range_2d(sPt2D const& first, sPt2D const& last, sPt_f const& pt_func)
        {
            auto pts = to_signed_range_2d(first, last);

            execute(pts, pt_func);
        }


        void for_each_in_range_2d(uInt width, uInt height, uXY_f const& xy_func)
        {
            auto const pt_func = [&](auto const& pt){ return xy_func(pt.x, pt.y); };

            for_each_in_range_2d(width, height, pt_func);
        }


        void for_each_in_range_2d(uInt x_begin, uInt x_end, uInt y_begin, uInt y_end, uXY_f const& xy_func)
        {
            auto const pt_func = [&](auto const& pt){ return xy_func(pt.x, pt.y); };
            
            auto pts = to_unsigned_range_2d(x_begin, x_end, y_begin, y_end);

            execute(pts, pt_func);
        }


        void for_each_in_range_2d(sInt x_begin, sInt x_end, sInt y_begin, sInt y_end, sXY_f const& xy_func)
        {
            auto const pt_func = [&](auto const& pt){ return xy_func(pt.x, pt.y); };

            auto pts = to_signed_range_2d(x_begin, x_end, y_begin, y_end);

            execute(pts, pt_func);
        }


        
    }


    namespace par
    {
        static void execute(UnsignedRange& ids, uInt_f const& id_func)
        {
            std::for_each(std::execution::par, ids.begin(), ids.end(), id_func);
        }


        static void execute(SignedRange& ids, sInt_f const& id_func)
        {
            std::for_each(std::execution::par, ids.begin(), ids.end(), id_func);
        }


        static void execute(UnsignedPointRange2D& pts, uPt_f const& pt_func)
        {
            auto const r_pt_func = [&](auto const& pt) { pt_func({ pt.x, pt.y }); };

            std::for_each(std::execution::par, pts.begin(), pts.end(), r_pt_func);
        }


        static void execute(SignedPointRange2D& pts, sPt_f const& pt_func)
        {
            auto const r_pt_func = [&](auto const& pt) { pt_func({ pt.x, pt.y }); };

            std::for_each(std::execution::par, pts.begin(), pts.end(), r_pt_func);
        }



        void for_each_in_range(uInt size, uInt_f const& id_func)
        {
            UnsignedRange ids(size);
            execute(ids, id_func);
        }
        

        void for_each_in_range(uInt begin, uInt end, uInt_f const& id_func)
        {
            UnsignedRange ids(begin, end);
            execute(ids, id_func);
        }


        void for_each_in_range(sInt begin, sInt end, sInt_f const& id_func)
        {
            SignedRange ids(begin, end);
            execute(ids, id_func);
        }


        void for_each_in_range_2d(uInt width, uInt height, uPt_f const& pt_func)
        {
            UnsignedPointRange2D pts(width, height);

            execute(pts, pt_func);
        }


        void for_each_in_range_2d(uPt2D const& first, uPt2D const& last, uPt_f const& pt_func)
        {
            auto pts = to_unsigned_range_2d(first, last);

            execute(pts, pt_func);
        }


        void for_each_in_range_2d(sPt2D const& first, sPt2D const& last, sPt_f const& pt_func)
        {
            auto pts = to_signed_range_2d(first, last);

            execute(pts, pt_func);
        }


        void for_each_in_range_2d(uInt width, uInt height, uXY_f const& xy_func)
        {
            auto const pt_func = [&](auto const& pt){ return xy_func(pt.x, pt.y); };

            for_each_in_range_2d(width, height, pt_func);
        }


        void for_each_in_range_2d(uInt x_begin, uInt x_end, uInt y_begin, uInt y_end, uXY_f const& xy_func)
        {
            auto const pt_func = [&](auto const& pt){ return xy_func(pt.x, pt.y); };
            
            auto pts = to_unsigned_range_2d(x_begin, x_end, y_begin, y_end);

            execute(pts, pt_func);
        }


        void for_each_in_range_2d(sInt x_begin, sInt x_end, sInt y_begin, sInt y_end, sXY_f const& xy_func)
        {
            auto const pt_func = [&](auto const& pt){ return xy_func(pt.x, pt.y); };

            auto pts = to_signed_range_2d(x_begin, x_end, y_begin, y_end);

            execute(pts, pt_func);
        }



    }
}




























