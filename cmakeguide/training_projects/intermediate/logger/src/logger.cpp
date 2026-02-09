#include "logger.h"
#include <iostream>
#include <ctime>
#include <sstream>

namespace logger {
    Logger::Logger(const std::string& filename) 
        : current_level_(Level::INFO), filename_(filename) {
        open_file();
    }
    
    Logger::~Logger() {
        if (file_.is_open()) {
            file_.close();
        }
    }
    
    void Logger::log(Level level, const std::string& message) {
        if (level < current_level_) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Get current time
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        
        // Format log entry
        std::ostringstream oss;
        oss << "[" << time_str << "] [" << level_to_string(level) << "] " << message << std::endl;
        
        // Write to file and console
        file_ << oss.str();
        file_.flush();
        
        if (level >= Level::WARNING) {
            std::cerr << oss.str();
        } else {
            std::cout << oss.str();
        }
    }
    
    void Logger::debug(const std::string& message) {
        log(Level::DEBUG, message);
    }
    
    void Logger::info(const std::string& message) {
        log(Level::INFO, message);
    }
    
    void Logger::warning(const std::string& message) {
        log(Level::WARNING, message);
    }
    
    void Logger::error(const std::string& message) {
        log(Level::ERROR, message);
    }
    
    void Logger::set_level(Level level) {
        current_level_ = level;
    }
    
    void Logger::set_filename(const std::string& filename) {
        filename_ = filename;
        if (file_.is_open()) {
            file_.close();
        }
        open_file();
    }
    
    std::string Logger::level_to_string(Level level) {
        switch (level) {
            case Level::DEBUG:   return "DEBUG";
            case Level::INFO:    return "INFO";
            case Level::WARNING: return "WARNING";
            case Level::ERROR:   return "ERROR";
            default:             return "UNKNOWN";
        }
    }
    
    void Logger::open_file() {
        file_.open(filename_, std::ios::app);
        if (!file_.is_open()) {
            std::cerr << "Failed to open log file: " << filename_ << std::endl;
        }
    }
}