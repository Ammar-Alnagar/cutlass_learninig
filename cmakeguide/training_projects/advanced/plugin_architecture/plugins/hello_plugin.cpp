#include "../include/plugin_interface.h"
#include <string>

class HelloPlugin : public PluginInterface {
public:
    std::string name() const override {
        return "HelloPlugin";
    }
    
    std::string version() const override {
        return "1.0.0";
    }
    
    std::string execute(const std::string& input) override {
        return "Hello, " + input + "!";
    }
};

// Export the function to create the plugin
PLUGIN_EXPORT PluginInterface* create_plugin() {
    return new HelloPlugin();
}