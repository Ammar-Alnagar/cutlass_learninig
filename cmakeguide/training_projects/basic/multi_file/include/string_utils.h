#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <string>

namespace utils {
    std::string to_upper(const std::string& str);
    std::string reverse_string(const std::string& str);
    int count_words(const std::string& str);
}

#endif