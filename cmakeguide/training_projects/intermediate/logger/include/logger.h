#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <mutex>

namespace logger {
    enum class Level {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };
    
    class Logger {
    public:
        explicit Logger(const std::string& filename = "app.log");
        ~Logger();
        
        void log(Level level, const std::string& message);
        void debug(const std::string& message);
        void info(const std::string& message);
        void warning(const std::string& message);
        void error(const std::string& message);
        
        void set_level(Level level);
        void set_filename(const std::string& filename);
        
    private:
        std::string level_to_string(Level level);
        std::ofstream file_;
        Level current_level_;
        std::mutex mutex_;
        std::string filename_;
        
        void open_file();
    };
}

#endif