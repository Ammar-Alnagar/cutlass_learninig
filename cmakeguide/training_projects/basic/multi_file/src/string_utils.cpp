#include "string_utils.h"
#include <algorithm>
#include <cctype>

namespace utils {
    std::string to_upper(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::toupper);
        return result;
    }
    
    std::string reverse_string(const std::string& str) {
        std::string result = str;
        std::reverse(result.begin(), result.end());
        return result;
    }
    
    int count_words(const std::string& str) {
        if (str.empty()) return 0;
        
        int count = 0;
        bool in_word = false;
        
        for (char c : str) {
            if (std::isspace(c)) {
                in_word = false;
            } else if (!in_word) {
                in_word = true;
                count++;
            }
        }
        
        return count;
    }
}