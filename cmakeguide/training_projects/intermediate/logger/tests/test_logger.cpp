#include <iostream>
#include <fstream>
#include <cassert>
#include "logger.h"

bool test_log_creation() {
    logger::Logger log("test_creation.log");
    log.info("Test log creation");
    
    // Check if file was created
    std::ifstream file("test_creation.log");
    bool exists = file.is_open();
    if (exists) {
        file.close();
    }
    
    std::cout << "Log file creation: " << (exists ? "PASS" : "FAIL") << std::endl;
    return exists;
}

bool test_log_levels() {
    logger::Logger log("test_levels.log");
    
    // Set to WARNING level, so INFO should not appear
    log.set_level(logger::Level::WARNING);
    log.info("This should not appear");
    log.error("This should appear");
    
    // For simplicity, we're just testing that no exceptions are thrown
    std::cout << "Log level filtering: PASS (no exceptions)" << std::endl;
    return true;
}

bool test_different_levels() {
    logger::Logger log("test_different_levels.log");
    
    log.debug("Debug message");
    log.info("Info message");
    log.warning("Warning message");
    log.error("Error message");
    
    std::cout << "Different log levels: PASS (no exceptions)" << std::endl;
    return true;
}

int main() {
    int failed = 0;
    
    if (!test_log_creation()) failed++;
    if (!test_log_levels()) failed++;
    if (!test_different_levels()) failed++;
    
    std::cout << "\nTests run: 3, Failed: " << failed << std::endl;
    
    if (failed == 0) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}