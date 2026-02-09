# CMake and Make Cheat Sheet

## Essential CMake Commands

### Project Setup
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
```

### Create Targets
```cmake
# Executable
add_executable(app src/main.cpp)

# Library
add_library(mylib src/lib.cpp)

# Link
target_link_libraries(app PRIVATE mylib)
```

### Include Directories
```cmake
target_include_directories(mylib PUBLIC include/)
```

### Find Packages
```cmake
find_package(PkgName REQUIRED)
target_link_libraries(app PRIVATE PkgName::PkgName)
```

## Essential Make Commands

### Basic Makefile
```makefile
CC = gcc
CFLAGS = -Wall -g
TARGET = app
SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean
```

### Common Make Commands
```bash
make                    # Build default target
make target            # Build specific target
make -j4               # Build with 4 parallel jobs
make clean             # Run clean target
make -f Makefile.debug # Use specific Makefile
make V=1               # Verbose output
```

## Quick CMake Patterns

### Library with Interface
```cmake
add_library(mylib INTERFACE)
target_include_directories(mylib INTERFACE include/)
```

### Conditional Compilation
```cmake
if(WIN32)
    target_compile_definitions(app PRIVATE WIN32_PLATFORM)
endif()
```

### Generator Expressions
```cmake
target_compile_options(app PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
)
```

## Quick Make Patterns

### Automatic Dependency Tracking
```makefile
CFLAGS += -MMD -MP
-include $(SOURCES:.c=.d)
```

### Directory Creation
```makefile
OBJDIR = obj
$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@
```

### Phony Targets
```makefile
.PHONY: all clean test install
```

## Common Build Commands

### CMake Workflow
```bash
# Out-of-source build
mkdir build && cd build
cmake ..                    # Configure
cmake --build .             # Build (or just 'make')
ctest                       # Run tests
make install                # Install
```

### CMake Options
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..     # Debug build
cmake -DCMAKE_BUILD_TYPE=Release ..   # Release build
cmake -DCMAKE_INSTALL_PREFIX=/path .. # Install prefix
cmake --list-targets                  # List available targets
```

## Platform Detection

### CMake
```cmake
if(WIN32)
    # Windows
elseif(APPLE)
    # macOS
elseif(UNIX)
    # Linux/Unix
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 64-bit
else()
    # 32-bit
endif()
```

### Make
```makefile
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    # Linux
endif
ifeq ($(UNAME_S),Darwin)
    # macOS
endif
```

## Error Diagnosis

### Common CMake Errors
- `"Could not find a package configuration file"` → Check `CMAKE_PREFIX_PATH`
- `"Target has no SOURCES property"` → Add sources to target
- `"Include directory not found"` → Use `target_include_directories`

### Common Make Errors
- `"Missing separator"` → Check for tab vs space indentation
- `"Nothing to be done"` → Check file timestamps
- `"Undefined reference"` → Check linking order

## Debug Commands

### CMake Debug
```bash
cmake --debug-output ..     # Verbose output
cmake -LAH ..               # List cached variables
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..  # Verbose makefile
```

### Make Debug
```bash
make -n                     # Dry run (don't execute)
make -p                     # Print database
make print-VARNAME          # Print variable value
make --dry-run --debug=b    # Detailed dry run
```

## Best Practices Summary

### CMake Best Practices
1. Use `target_*` commands instead of global ones
2. Always specify visibility (PRIVATE/PUBLIC/INTERFACE)
3. Use modern CMake (3.15+)
4. Use imported targets when available
5. Don't hardcode paths

### Make Best Practices
1. Use `.PHONY` for non-file targets
2. Use automatic variables (`$@`, `$<`, `$^`)
3. Use pattern rules for compilation
4. Use `+=` to append to variables
5. Use `@` to suppress command echo

## Quick Reference Commands

| Purpose | CMake | Make |
|---------|-------|------|
| Build | `cmake --build .` | `make` |
| Clean | `make clean` | `make clean` |
| Configure | `cmake ..` | N/A |
| Install | `make install` | `make install` |
| Run Tests | `ctest` | `make test` |
| Verbose | `-DCMAKE_VERBOSE_MAKEFILE=ON` | `make V=1` |
| Parallel | `-j4` | `-j4` |

## File Extensions
- CMake files: `CMakeLists.txt`, `.cmake`
- Make files: `Makefile`, `makefile`, `.mk`
- CMake cache: `CMakeCache.txt`
- Generated files: Usually in build directory