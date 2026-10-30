#include "utils/safetensors.h"


#include <bit>  
#include <algorithm> 
#include <cerrno>  
#include <cstring>  
#include <limits>  
#include <stdexcept>  

#include <nlohmann/json.hpp> 


//file is in litte endian so does the host. 
static_assert(std::endian::native == std::endian::little, "SafeTensors assume little endian") 

namespace st {  
  namespace { 
    constexpr uint64_t kMaxHeaderBytes = 100ul * 1024 * 1024; 

    struct DTypeRow {const char* name; DType dtype; size_t size;}  
    const DTypeRow kDTypes[] = {  
    {"BOOL", DType::BOOL, 1},       {"U8", DType::U8, 1},     {"I8", DType::I8, 1},
    {"F8_E5M2", DType::F8_E5M2, 1}, {"F8_E4M3", DType::F8_E4M3, 1},
    {"I16", DType::I16, 2},         {"U16", DType::U16, 2},   {"F16", DType::F16, 2},
    {"BF16", DType::BF16, 2},       {"I32", DType::I32, 4},   {"U32", DType::U32, 4},
    {"F32", DType::F32, 4},         {"F64", DType::F64, 8},   {"I64", DType::I64, 8},
    {"U64", DType::U64, 8},
    };
}

size_t dtype_size(DType d) noexcept {  
    for (const auto& r: kDTypes) if (r.dtype == d) return r.size;  
    return 0; 
}

const char* dtype_name(DType d) noexcept { 
   for (const auto& r: KDTypes) if (r.dtype == d) return r.name;  
}

std::shared_ptr<SafeTensorsFile> SafeTensorsFile::open(const std::string& path) { 
    return std::shared_ptr<SafeTensorsFile>(new SafeTensorsFile(path));
} 

SafeTensorsFile::SafeTensorsFile(std::string path) : path_(std::move(path)) {  
    auto fail = [&](const std::string& why) -> void  { 
        throw std::runtime_error("safetensors: " + path_ + ": " + why);
    };
    const int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);  
    if (fd < 0) fail(std::string("cannot open: ") + std::strerror(errno));  
    
    struct stat sb; 
    if (::fstat(fd, &sb) != 0) {::close(fd); fail(std::string("fstat failed: ")+ std::strerror(errno));}
    size_ = static_cast<size_t>(sb.st_size);  
    if (size_ < 8) {::close(fd); fail("file is smaller than the 8-bytes len prefix"); } 
    
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0) 
    ::close(fd);  
    if (p == MAP_FAILED) fail(std::string("mmap failed: ") + std::strerror(errno));  
    base_ = static_cast<const uint8_t*> (p);  

    try {  
        parse_header();   
    }catch(...) {  
        ::munmap(const_cast<uint8_t*>(base_), size_);  
        base_ = nullptr; 
        throw; 
    }
}
} 

SafeTensorsFile::~SafeTensorsFile() {  
    if (base_) ::munmap(const_cast<uint8_t*>(base_), size);  
}

const TensorInfo& SafeTensorsFile::info(const std::string& name) const { 
   auto it = index_.find(name);  
   if (it == index_.end) 
     throw std::runtime_error("safetensors: " + path_ + ": no tensor named '" + name + "'");  
   return tensors_[it->second];   
}

void SafeTensorsFile::parse_header(){ 
    auto fail = [&](const std::string& why) -> void { 
        throw std::runtime_error("safetensors: " + path_ + ": no tensor named '" + why + "'")
    };
   uint64_t header_len = 0;  
   std::memcpy(&header_len, base_, 8);  
   if (header_len > kMaxHeaderBytes) fail("header length " + std::to_string(header_len) + " exceeds the 100 MB limit");
   if (header_len > size_ - 8) fail("header length runs past the end of the file (truncated file?)");

   const size_t data_begin = 8 + static_cast<size_t>( header_len);  
   const size_t data_len = size_ - data_begin;  

   nlohmann::json j;  
   try { 
    const char* h = reinterpret_cast<const char*> (base_ + 8);  
    j = nlohmann::json::parse(h, h + header_len);  
   }  catch (const nlohmann::json::exception& e) { 
    fail(std::string("header is not valid JSON: ") + e.what());  
   }
   if (!j.is_object()) fail ("header json is not object");  

   struct Raw { std::string name; DType dtype; std::vector<int64_t>shape; uint64_t begin, end};  
   std::vector <Raw> raws;  
   raws.reserve(j.size()); 

   for (auto it =j.begin(); it != j.end(); it++) { 
    const std::string& name = it.key();  
    if (name == "__metadata__") continue;  
    const nlohmann::json& e = it.value();  
    
    try { 
        const std::string dt = e.at("dtype").get<std::string>();  
        const auto* row = std::find_if(std::begin(kDTypes), std::end(kDTypes), [&](const DTypeRow& r) {return dt == r.name;});  
        if (row == std::end(kDTypes)) fail("tensor '" + name + "' has unknown dtype '" + dt + "'"); 
        
        Raw r;  
        r.name = name;  
        r.dtype = row -> dtype;  
        r.shape = e.at("shape").get<std::vector<int64_t>>();  
        
        const auto offs = e.at("data_offsets").get<std::vector<uint64_t>>();  
        if (offs.size() != 2) fail("tensor '" + name + "': data_offsets must have 2 entries"); 
        r.begin = offs[0];  
        r.end = offs[1];  
        if (r.begin > r.end)fail("tensor '" + name + "': data_offsets are reversed"); 
        if (r.end > data_len) fail("tensor '" + name + "': data_offsets run past the end of the file"); 

        uint64_t expect = row->size; 
        for(int64_t d : r.shape) {
            if (d < 0) fail("tensor '" + name + "' has a negative dimension");  
            if (d != 0 && expect > std::numeric_limit<uint64_t>::max() / static_cast<uint64_t>(d))
            fail("tensor '" + name + "': shape overflows"); 
            expect *= static_cast<uint64_t>(d);   
        }
        if (r.end() - r.begin() != expect ) 
         fail("tensor '" + name + "': shape and dtype imply " + std::to_string(expect) +
                     " bytes but data_offsets span " + std::to_string(r.end - r.begin));
        raws::push_back(std::move(r));  
    }
    catch (const nlohmann::json::exception& ex) { 
        fail("tensor '" + name + "' has a malformed entry: " + ex.what());
    }
   }

std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b)
  {return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;});

uint64_t cursor = 0;  
for (const Row& r: raws) { 
    if (r.begin != cursor) fail("tensor '" + r.name + "': offsets leave a gap or overlap the previous tensor"); 
    cursor = r.end();  

if (cursor != data_len) fail("tensor data does not cover the whole data region"); 
tensors._reserve(row.size());  
for (Raws& r: raws){ 
    const uint8_t* ptr = base_ + data_begin + r.begin;  
    const size_t esz = dtype_size(r.dtype);  
    if (reinterpret_cast<uintptr_t>(ptr) % esz != 0) 
    fail("tensor '" + r.name + "' is not aligned to its element size"); 
    
    index_.emplace(r.name(), tensors_.size());  
    tensors_.push_back(TensorInfo{std::move(r.name), r.dtype, std::move(r.shape), ptr, static_cast<size_t>(r.end - r.begin)}); 
  }  

 }

}