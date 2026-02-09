#include <iostream>
#include "logger.h"

int main() {
    logger::Logger log("demo.log");
    
    std::cout << "Logger Demo" << std::endl;
    
    log.info("Application started");
    log.debug("Debug information");
    log.warning("This is a warning");
    log.error("An error occurred");
    
    // Change log level
    log.set_level(logger::Level::WARNING);
    log.info("This info message won't appear due to log level");
    log.error("This error will still appear");
    
    std::cout << "Check demo.log for logged messages" << std::endl;
    
    return 0;
}