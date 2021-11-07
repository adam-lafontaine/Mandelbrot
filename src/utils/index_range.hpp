#include <iterator>


class UnsignedRange
{
public:

    using index_type = size_t;

    class iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = index_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = value_type;

        value_type m_val;

        explicit iterator() : m_val(0) {}

        explicit iterator(value_type val) : m_val(val) {}
        
        reference operator*() const { return m_val; }
        iterator& operator++() { ++m_val; return *this; }
        iterator operator++(int) { iterator retval = *this; ++(*this); return retval; }
        bool operator==(iterator other) const { return m_val == other.m_val; }
        bool operator!=(iterator other) const { return !(*this == other); }        
    };

private:

    index_type m_min = 0;
    index_type m_max = 1;

    template<typename INT_T>
    index_type to_min(INT_T val) { return val < 0 ? 0 : static_cast<index_type>(val); }

    template<typename INT_T>
    index_type to_max(INT_T val) { return val < 1 ? 0 : static_cast<index_type>(val - 1); }

public:

    template<typename INT_T>
    UnsignedRange(INT_T size) : m_max(to_max(size)) {}

    template<typename INT_T>
    UnsignedRange(INT_T begin, INT_T end)
    {
        if(end >= begin)
        {
            m_min = to_min(begin);
            m_max = to_max(end);
        }
        else
        {
            m_min = to_min(end);
            m_max = to_max(begin);
        }        
    }

    iterator begin() { return iterator(m_min); }
    iterator end() { return iterator(m_max + 1); }
};



class SignedRange
{
public:

    using index_type = long long;

    class iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = index_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = value_type;

        value_type m_val;

        explicit iterator() : m_val(0) {}

        explicit iterator(value_type val) : m_val(val) {}
        
        reference operator*() const { return m_val; }
        iterator& operator++() { ++m_val; return *this; }
        iterator operator++(int) { iterator retval = *this; ++(*this); return retval; }
        bool operator==(iterator other) const { return m_val == other.m_val; }
        bool operator!=(iterator other) const { return !(*this == other); }        
    };

private:

    index_type m_min = 0;
    index_type m_max = 1;

    template<typename INT_T>
    index_type to_min(INT_T val) { return static_cast<index_type>(val); }

    template<typename INT_T>
    index_type to_max(INT_T val) { return static_cast<index_type>(val - 1); }

public:

    template<typename INT_T>
    SignedRange(INT_T begin, INT_T end)
    {
        if(end >= begin)
        {
            m_min = to_min(begin);
            m_max = to_max(end);
        }
        else
        {
            m_min = to_min(end);
            m_max = to_max(begin);
        }
    }

    iterator begin() { return iterator(m_min); }
    iterator end() { return iterator(m_max + 1); }
};