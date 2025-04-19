#define IMAGE_READ
#include "../image/image.hpp"

#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <algorithm>
#include <functional>
#include <cassert>


using ByteBuffer = MemoryBuffer<u8>;
namespace mb = memory_buffer;
namespace fs = std::filesystem;


namespace a2b
{
    
    namespace img = image;
    using p32 = img::Pixel;
    

    template <typename T>
    using fn = std::function<T>;
}


namespace a2b
{
namespace internal
{
    static ByteBuffer read_bytes(fs::path const& path)
    {
        ByteBuffer buffer;

        auto size = fs::file_size(path);
        if (size == 0 || !mb::create_buffer(buffer, size, ""))
        {
            assert(false);
            return buffer;
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            assert(false);
            mb::destroy_buffer(buffer);
            return buffer;
        }

        file.read((char*)buffer.data_, size);
        buffer.size_ = size;

        file.close();
        return buffer;
    }


    static bool write_cpp_file(std::string code_str, fs::path const& dst_dir, std::string const& tag)
    {
        auto filename = tag + ".cpp";
        auto filepath = fs::path(dst_dir) / filename;
    
        std::ofstream file(filepath);
    
        if (!file.is_open())
        {
            return false;
        }

        file << "// " << filename << "\n\n";    
        file << code_str;
    
        file.close();
    
        return true;
    }
    

    class DataSize    
    {
    public:
        std::string name;
        u32 size;

        DataSize(std::string&& n, u32 s)
        {
            name = std::move(n);
            size = s;
        }
    };
}
}


/* files */

namespace a2b
{
namespace internal
{
    class DirFiles
    {
    public:
        std::string dir_name;

        std::vector<DataSize> files;
    };


    static DirFiles read_files(fs::path const& dir, std::ofstream& bin)
    {
        DirFiles df;

        df.dir_name = dir.filename().string();

        for (auto const& entry : fs::directory_iterator(dir))
        {            
            auto p = entry.path();
            if (!fs::is_regular_file(p))
            {
                continue;
            }

            auto buffer = read_bytes(p);

            df.files.emplace_back(p.stem().string(), buffer.size_);
            
            bin.write((char*)buffer.data_, buffer.size_);

            mb::destroy_buffer(buffer);
        }

        return df;
    }


    static bool write_file_sizes(std::vector<DirFiles> const& data, fs::path const& dst_dir, std::string const& name)
    {
        auto tab = "    ";
        auto tabtab = "        ";
        std::ostringstream oss;

        oss
        << "const struct {\n\n";

        for (auto const& item : data)
        {
            oss
            << tab << "struct {\n";

            for (auto const& f : item.files)
            {
                oss
                << tabtab << "struct { " << "unsigned size; unsigned offset; } " << f.name << ";\n";
            }

            oss
            << tab << "} " << item.dir_name << ";\n\n";
        }

        u32 offset = 0;

        oss
        << "}\n"
        << name << " = \n"
        << "{\n";

        for (auto const& item : data)
        {
            oss
            << tab << "{\n";

            for (auto const& f : item.files)
            {
                oss
                << tabtab << "{ " << f.size << ", " << offset << " },\n";

                offset += f.size;
            }

            oss
            << tab << "},\n";
        }

        oss
        << "};\n";

        return internal::write_cpp_file(oss.str(), dst_dir, name);
    }
}
}


namespace a2b
{
    class AssetFiles
    {
    public:

        std::vector<internal::DirFiles> file_sizes;
        std::ofstream bin_file;
        fs::path out_dir;
    };


    inline bool create(AssetFiles& af, fs::path const& out_dir, cstr bin_name)
    {
        auto bin_out = out_dir / (std::string(bin_name) + ".bin");

        af.bin_file = std::ofstream(bin_out);
        if (!af.bin_file.is_open())
        {
            return false;
        }

        af.out_dir = out_dir;

        return true;
    }


    inline bool append_file_dir(AssetFiles& af, fs::path const& dir)
    {
        if (!fs::is_directory(dir))
        {
            return false;
        }

        af.file_sizes.push_back(internal::read_files(dir, af.bin_file));

        return true;
    }


    inline bool save_and_close(AssetFiles& af, cstr size_file_name)
    {
        af.bin_file.close();

        return internal::write_file_sizes(af.file_sizes, af.out_dir, size_file_name);
    }
}