#include "plugin_interface.h"
#include <iostream>
#include <vector>
#include <memory>
#include <dlfcn.h> // For dynamic loading on Unix-like systems

class PluginManager {
public:
    PluginManager() = default;
    ~PluginManager() = default;
    
    bool load_plugin(const std::string& path) {
#ifdef _WIN32
        HMODULE handle = LoadLibrary(path.c_str());
        if (!handle) {
            std::cerr << "Could not load plugin: " << path << std::endl;
            return false;
        }
#else
        void* handle = dlopen(path.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Could not load plugin: " << dlerror() << std::endl;
            return false;
        }
#endif
        
        // Store the handle for later cleanup
        handles_.push_back(handle);
        
        // Get the create function
#ifdef _WIN32
        CreatePluginFunc create_func = (CreatePluginFunc)GetProcAddress(handle, "create_plugin");
#else
        CreatePluginFunc create_func = (CreatePluginFunc)dlsym(handle, "create_plugin");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            std::cerr << "Could not load symbol create_plugin: " << dlsym_error << std::endl;
            return false;
        }
#endif
        
        if (!create_func) {
            std::cerr << "Could not find create_plugin function in: " << path << std::endl;
            return false;
        }
        
        // Create the plugin instance
        PluginInterface* plugin = create_func();
        if (!plugin) {
            std::cerr << "Could not create plugin instance from: " << path << std::endl;
            return false;
        }
        
        plugins_.push_back(std::unique_ptr<PluginInterface>(plugin));
        return true;
    }
    
    void execute_all_plugins(const std::string& input) {
        for (auto& plugin : plugins_) {
            std::cout << "Executing " << plugin->name() 
                      << " v" << plugin->version() << ": ";
            std::string result = plugin->execute(input);
            std::cout << result << std::endl;
        }
    }
    
private:
    std::vector<std::unique_ptr<PluginInterface>> plugins_;
#ifdef _WIN32
    std::vector<HMODULE> handles_;
#else
    std::vector<void*> handles_;
#endif
};