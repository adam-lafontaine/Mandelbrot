#define IMAGE_READ
#include "../image/image.hpp"

#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <algorithm>

using ByteBuffer = MemoryBuffer<u8>;
namespace mb = memory_buffer;
namespace fs = std::filesystem;

namespace i2c
{
    
    namespace img = image;

    using p32 = img::Pixel;

    template <typename T>
    using fn = std::function<T>;

    using convert = fn<ByteBuffer(ByteBuffer const&)>;


namespace internal
{
    static bool is_file(fs::path const& entry)
    {
        return fs::is_regular_file(entry) && entry.has_filename() && entry.has_extension();
    }
    
    
    static std::string to_struct_str(img::ImageGray image, std::string const& name)
    {
        cstr tab = "    ";
        cstr tabtab = "        ";
        auto size = image.width * image.height;

        std::ostringstream oss;

        oss
        //<< "static constexpr struct\n"
        << "const struct\n"
        << "{\n";

        oss
        << tab << "unsigned width;\n"
        << tab << "unsigned height;\n"
        << tab << "unsigned char data[" << size << "];\n";

        oss
        << "}\n"
        << name << " = \n"
        << "{\n";

        oss
        << tab << image.width << ",\n"
        << tab << image.height << ",\n"
        << tab << "{\n";

        u32 n = 0;
        oss << tabtab;

        for (u32 i = 0; i < size; i++)
        {
            if (n == image.width)
            {
                n = 0;
                oss << "\n" << tabtab;
            }

            oss << (int)image.data_[i] << ", ";
            ++n;
        }

        oss
        << "\n"
        << tab << "}\n";

        oss
        << "};\n";

        return oss.str();
    }


    static bool write_cpp_file(std::string code_str, fs::path const& dst_dir, std::string const& name)
    {
        auto filename = name + ".cpp";
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


    static bool validate_directories(fs::path const& src_dir, fs::path const& dst_dir)
    {
        printf("check src directory: ");
        if (!fs::exists(src_dir) || !fs::is_directory(src_dir))
        {
            printf("FAIL\n");
            return false;
        }
        printf("OK\n");

        printf("check dst directory: ");
        if (!fs::exists(dst_dir) || !fs::is_directory(dst_dir))
        {
            printf("FAIL\n");
            return false;
        }
        printf("OK\n");
        
        return true;
    }


    static bool image_to_code(fs::path const& filepath, fs::path const& dst_dir, fn<u8(p32)> const& encode)
    {
        auto name = filepath.stem().string();

        img::Image img32;
        img::ImageGray img8;

        auto const cleanup = [&]()
        {
            img::destroy_image(img32);
            img::destroy_image(img8);
        };
        
        printf("read image %s ", name.c_str());
        if (!img::read_image_from_file(filepath.string().c_str(), img32))
        {
            printf("FAIL\n");
            cleanup();
            return false;
        }
        printf("OK\n");

        auto w = img32.width;
        auto h = img32.height;

        printf("create gray ");
        if (!img::create_image(img8, w, h))
        {
            printf("FAIL\n");
            cleanup();
            return false;
        }
        printf("OK\n");

        printf("encode image ");
        img::transform(img::make_view(img32), img::make_view(img8), encode);
        printf("OK\n");
        
        printf("write cpp ");
        auto str = to_struct_str(img8, name);
        if (!write_cpp_file(str, dst_dir, name))
        {
            printf("FAIL\n");
            cleanup();
            return false;
        }
        printf("OK\n");

        cleanup();
        return true;
    }


    static bool images_to_code(fs::path const& src_dir, fs::path const& dst_dir, fn<u8(p32)> const& encode)
    {  
        for (auto const& entry : fs::directory_iterator(src_dir))
        {
            if (!internal::is_file(entry))
            {
                continue;
            }

            auto path = entry.path();

            printf("%s\n", path.string().c_str());
            image_to_code(path, dst_dir, encode);
        }

        return true;
    }


    static bool image_dimensions_to_code(fs::path const& src_dir, fs::path const& dst_dir, std::string const& name)
    {
        auto tab = "    ";

        std::vector<u32> widths;
        std::vector<u32> heights;
        std::vector<std::string> names;

        for (auto const& entry : fs::directory_iterator(src_dir))
        {
            if (!internal::is_file(entry))
            {
                continue;
            }

            auto path = entry.path();
            img::Image img32;
            if (!img::read_image_from_file(path.string().c_str(), img32))
            {
                continue;
            }
            
            widths.push_back(img32.width);
            heights.push_back(img32.height);
            names.push_back(path.stem().string());

            img::destroy_image(img32);
        }

        auto count = names.size();

        std::ostringstream oss;
        oss
        << "const struct\n"
        << "{\n";

        for (u32 i = 0; i < count; i++)
        {
            oss << tab << "struct { unsigned width; unsigned height; } " << names[i] << ";\n";
        }

        oss
        << "}\n"
        << name << " = \n"
        << "{\n";

        for (u32 i = 0; i < count; i++)
        {
            oss << tab << "{ " << widths[i] << ", " << heights[i] << " },\n";
        }

        oss
        << "};\n";

        return write_cpp_file(oss.str(), dst_dir, name);
    }
}
}

namespace i2c
{
    inline bool image_to_code(fs::path const& filepath, fs::path const& dst_dir, fn<u8(p32)> const& encode)
    {
        printf("check dst directory: ");
        if (!fs::exists(dst_dir) || !fs::is_directory(dst_dir))
        {
            printf("FAIL\n");
            return false;
        }
        printf("OK\n");

        return internal::image_to_code(filepath, dst_dir, encode);
    }
    
    
    inline bool images_to_code(fs::path const& src_dir, fs::path const& dst_dir, fn<u8(p32)> const& encode)
    {
        if (!internal::validate_directories(src_dir, dst_dir))
        {
            return false;
        }

        return internal::images_to_code(src_dir, dst_dir, encode);
    }


    inline bool image_dimensions_to_code(fs::path const& src_dir, fs::path const& dst_dir, cstr name)
    {
        if (!internal::validate_directories(src_dir, dst_dir))
        {
            return false;
        }

        return internal::image_dimensions_to_code(src_dir, dst_dir, name);
    }
}