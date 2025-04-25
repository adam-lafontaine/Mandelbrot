#include <functional>


namespace for_each_in_range
{
    using uInt = unsigned long;
    using sInt = long;

    using uInt_f = std::function<void(uInt id)>;
    using sInt_f = std::function<void(sInt id)>;

    using uXY_f = std::function<void(uInt x, uInt y)>;
    using sXY_f = std::function<void(sInt x, sInt y)>; 

    typedef struct
    {
        uInt x;
        uInt y;
    } uPt2D;

    typedef struct
    {
        sInt x;
        sInt y;
    } sPt2D;

    using uPt_f = std::function<void(uPt2D const&)>;
    using sPt_f = std::function<void(sPt2D const&)>;


    namespace seq
    {
        void for_each_in_range(uInt size, uInt_f const& id_func);

        void for_each_in_range(uInt begin, uInt end, uInt_f const& id_func);        

        void for_each_in_range(sInt begin, sInt end, sInt_f const& id_func);



        void for_each_in_range_2d(uInt width, uInt height, uPt_f const& pt_func);

        void for_each_in_range_2d(uPt2D const& first, uPt2D const& last, uPt_f const& pt_func);

        void for_each_in_range_2d(sPt2D const& first, sPt2D const& last, sPt_f const& pt_func);



        void for_each_in_range_2d(uInt width, uInt height, uXY_f const& xy_func);

        void for_each_in_range_2d(uInt x_begin, uInt x_end, uInt y_begin, uInt y_end, uXY_f const& xy_func);

        void for_each_in_range_2d(sInt x_begin, sInt x_end, sInt y_begin, sInt y_end, sXY_f const& xy_func);
        
    }


    namespace par
    {
        void for_each_in_range(uInt size, uInt_f const& id_func);

        void for_each_in_range(uInt begin, uInt end, uInt_f const& id_func);        

        void for_each_in_range(sInt begin, sInt end, sInt_f const& id_func);



        void for_each_in_range_2d(uInt width, uInt height, uPt_f const& pt_func);

        void for_each_in_range_2d(uPt2D const& first, uPt2D const& last, uPt_f const& pt_func);

        void for_each_in_range_2d(sPt2D const& first, sPt2D const& last, sPt_f const& pt_func);



        void for_each_in_range_2d(uInt width, uInt height, uXY_f const& xy_func);

        void for_each_in_range_2d(uInt x_begin, uInt x_end, uInt y_begin, uInt y_end, uXY_f const& xy_func);

        void for_each_in_range_2d(sInt x_begin, sInt x_end, sInt y_begin, sInt y_end, sXY_f const& xy_func);
    }
}
