#include "spatial/core/spat_format.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cstring>
#include <stdexcept>

namespace spatial {

namespace {

// Minimal, strict, zero-dependency JSON parser for .spat header
struct JsonParser {
    const std::string& s;
    size_t pos = 0;

    void skip_ws() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    }

    char peek() {
        skip_ws();
        return pos < s.size() ? s[pos] : '\0';
    }

    void expect(char c) {
        skip_ws();
        if (pos >= s.size() || s[pos] != c) {
            throw std::runtime_error(std::string("JSON parse error: expected '") + c +
                                     "' at offset " + std::to_string(pos));
        }
        ++pos;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') return out;
            if (c == '\\' && pos < s.size()) {
                char e = s[pos++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: out.push_back(e); break;
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("JSON unterminated string");
    }

    std::string parse_number_token() {
        skip_ws();
        size_t start = pos;
        if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) ++pos;
        while (pos < s.size() &&
               (std::isdigit(static_cast<unsigned char>(s[pos])) ||
                s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' ||
                s[pos] == '-' || s[pos] == '+')) {
            ++pos;
        }
        return s.substr(start, pos - start);
    }

    int64_t parse_int() {
        return std::stoll(parse_number_token());
    }

    uint64_t parse_uint() {
        return std::stoull(parse_number_token());
    }

    std::string parse_scalar_as_string() {
        char c = peek();
        if (c == '"') return parse_string();
        if (c == 't') { pos += 4; return "true"; }
        if (c == 'f') { pos += 5; return "false"; }
        if (c == 'n') { pos += 4; return "null"; }
        if (c == '[') {
            // Serialize simple arrays as comma-separated string for metadata
            expect('[');
            std::string out;
            bool first = true;
            while (peek() != ']') {
                if (!first) { expect(','); out += ","; }
                first = false;
                out += parse_scalar_as_string();
            }
            expect(']');
            return out;
        }
        return parse_number_token();
    }

    void skip_value() {
        char c = peek();
        if (c == '"') { (void)parse_string(); }
        else if (c == '{') {
            expect('{');
            bool first = true;
            while (peek() != '}') {
                if (!first) expect(',');
                first = false;
                (void)parse_string();
                expect(':');
                skip_value();
            }
            expect('}');
        } else if (c == '[') {
            expect('[');
            bool first = true;
            while (peek() != ']') {
                if (!first) expect(',');
                first = false;
                skip_value();
            }
            expect(']');
        } else {
            (void)parse_scalar_as_string();
        }
    }
};

}  // namespace

SpatFile::~SpatFile() {
    if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
        munmap(mmap_ptr_, mmap_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

SpatFile::SpatFile(SpatFile&& o) noexcept
    : fd_(o.fd_), mmap_ptr_(o.mmap_ptr_), mmap_size_(o.mmap_size_),
      data_off_(o.data_off_), data_size_(o.data_size_),
      header_json_(std::move(o.header_json_)),
      tensors_(std::move(o.tensors_)),
      meta_(std::move(o.meta_)) {
    o.fd_ = -1;
    o.mmap_ptr_ = nullptr;
    o.mmap_size_ = 0;
}

SpatFile& SpatFile::operator=(SpatFile&& o) noexcept {
    if (this != &o) {
        if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) munmap(mmap_ptr_, mmap_size_);
        if (fd_ >= 0) close(fd_);
        fd_ = o.fd_;
        mmap_ptr_ = o.mmap_ptr_;
        mmap_size_ = o.mmap_size_;
        data_off_ = o.data_off_;
        data_size_ = o.data_size_;
        header_json_ = std::move(o.header_json_);
        tensors_ = std::move(o.tensors_);
        meta_ = std::move(o.meta_);
        o.fd_ = -1;
        o.mmap_ptr_ = nullptr;
        o.mmap_size_ = 0;
    }
    return *this;
}

SpatFile SpatFile::open(const std::string& path) {
    SpatFile f;
    f.fd_ = ::open(path.c_str(), O_RDONLY);
    if (f.fd_ < 0) {
        throw std::runtime_error("SpatFile::open failed: " + path);
    }
    struct stat st{};
    if (fstat(f.fd_, &st) != 0 || st.st_size < 32) {
        throw std::runtime_error("SpatFile invalid size: " + path);
    }
    f.mmap_size_ = static_cast<size_t>(st.st_size);
    f.mmap_ptr_ = mmap(nullptr, f.mmap_size_, PROT_READ, MAP_PRIVATE, f.fd_, 0);
    if (f.mmap_ptr_ == MAP_FAILED) {
        f.mmap_ptr_ = nullptr;
        throw std::runtime_error("SpatFile mmap failed: " + path);
    }

    const auto* base = static_cast<const uint8_t*>(f.mmap_ptr_);
    if (std::memcmp(base, "SPAT\0\0\0\0", 8) != 0) {
        throw std::runtime_error("SpatFile bad magic: " + path);
    }
    uint32_t version = 0, flags = 0;
    uint64_t json_size = 0, data_off = 0;
    std::memcpy(&version, base + 8, 4);
    std::memcpy(&flags, base + 12, 4);
    std::memcpy(&json_size, base + 16, 8);
    std::memcpy(&data_off, base + 24, 8);
    (void)flags;

    if (version != 1) {
        throw std::runtime_error("SpatFile unsupported version: " + std::to_string(version));
    }
    if (32 + json_size > data_off || data_off > f.mmap_size_) {
        throw std::runtime_error("SpatFile corrupt header offsets");
    }

    f.data_off_  = static_cast<size_t>(data_off);
    f.data_size_ = f.mmap_size_ - f.data_off_;
    f.header_json_.assign(reinterpret_cast<const char*>(base + 32), json_size);

    JsonParser jp{f.header_json_, 0};
    jp.expect('{');
    bool first_top = true;
    while (jp.peek() != '}') {
        if (!first_top) jp.expect(',');
        first_top = false;
        std::string key = jp.parse_string();
        jp.expect(':');
        if (key == "metadata") {
            jp.expect('{');
            bool first_m = true;
            while (jp.peek() != '}') {
                if (!first_m) jp.expect(',');
                first_m = false;
                std::string mk = jp.parse_string();
                jp.expect(':');
                f.meta_[mk] = jp.parse_scalar_as_string();
            }
            jp.expect('}');
        } else if (key == "tensors") {
            jp.expect('[');
            bool first_t = true;
            while (jp.peek() != ']') {
                if (!first_t) jp.expect(',');
                first_t = false;
                SpatTensorInfo ti;
                jp.expect('{');
                bool first_f = true;
                while (jp.peek() != '}') {
                    if (!first_f) jp.expect(',');
                    first_f = false;
                    std::string fk = jp.parse_string();
                    jp.expect(':');
                    if (fk == "name") ti.name = jp.parse_string();
                    else if (fk == "dtype") ti.dtype = dtype_from_name(jp.parse_string());
                    else if (fk == "offset") ti.offset = jp.parse_uint();
                    else if (fk == "nbytes") ti.nbytes = jp.parse_uint();
                    else if (fk == "shape") {
                        jp.expect('[');
                        bool first_d = true;
                        while (jp.peek() != ']') {
                            if (!first_d) jp.expect(',');
                            first_d = false;
                            ti.shape.push_back(jp.parse_int());
                        }
                        jp.expect(']');
                    } else {
                        jp.skip_value();
                    }
                }
                jp.expect('}');
                f.tensors_[ti.name] = std::move(ti);
            }
            jp.expect(']');
        } else {
            jp.skip_value();
        }
    }
    jp.expect('}');

    return f;
}

bool SpatFile::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const SpatTensorInfo& SpatFile::info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("SpatFile tensor not found: " + name);
    }
    return it->second;
}

std::string SpatFile::meta_string(const std::string& key, const std::string& def) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? def : it->second;
}

int64_t SpatFile::meta_int(const std::string& key, int64_t def) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? def : std::stoll(it->second);
}

double SpatFile::meta_float(const std::string& key, double def) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? def : std::stod(it->second);
}

void* SpatFile::data_ptr() const {
    if (!mmap_ptr_) return nullptr;
    return static_cast<uint8_t*>(mmap_ptr_) + data_off_;
}

TensorView SpatFile::get_tensor(const std::string& name, const Buffer& data_buf) const {
    const auto& ti = info(name);
    Buffer sub = data_buf.sub(ti.offset, ti.nbytes);
    return TensorView(sub, ti.dtype, ti.shape);
}

}  // namespace spatial
