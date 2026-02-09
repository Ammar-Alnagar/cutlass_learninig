#include "../include/plugin_interface.h"
#include <string>
#include <algorithm>
#include <cctype>

class UpperCasePlugin : public PluginInterface {
public:
    std::string name() const override {
        return "UpperCasePlugin";
    }
    
    std::string version() const override {
        return "1.0.0";
    }
    
    std::string execute(const std::string& input) override {
        std::string result = input;
        std::transform(result.begin(), result.end(), result.begin(), ::toupper);
        return result;
    }
};

// Export the function to create the plugin
PLUGIN_EXPORT PluginInterface* create_plugin() {
    return new UpperCasePlugin();
}