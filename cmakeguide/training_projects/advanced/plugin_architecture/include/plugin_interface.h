#ifndef PLUGIN_INTERFACE_H
#define PLUGIN_INTERFACE_H

#include <string>

// Virtual base class for plugins
class PluginInterface {
public:
    virtual ~PluginInterface() = default;
    
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual std::string execute(const std::string& input) = 0;
};

// Macro to export plugin symbols on Windows
#ifdef _WIN32
    #define PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
    #define PLUGIN_EXPORT extern "C"
#endif

// Function type for plugin creation
typedef PluginInterface* (*CreatePluginFunc)();

#endif