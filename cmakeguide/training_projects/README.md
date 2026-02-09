# CMake and Make Training Projects

This directory contains hands-on training projects designed to help you master CMake and Make build systems through practical examples.

## Directory Structure

### `/basic`
Contains foundational projects for beginners:
- `hello_world/` - Simplest possible CMake project
- `library_example/` - Creating and using libraries
- `multi_file/` - Projects with multiple source files
- `simple_make/` - Pure Makefile example

### `/intermediate`
Projects that build on basic concepts:
- `calculator/` - Multi-file application with tests
- `logger/` - Custom library with threading and file I/O
- `data_processor/` - Data processing with configuration
- `network_client/` - Application with external dependencies

### `/advanced`
Complex projects demonstrating advanced concepts:
- `plugin_architecture/` - Dynamic plugin system
- `cross_platform_lib/` - Multi-platform library
- `cmake_package/` - Creating distributable packages
- `performance_benchmark/` - Performance optimization

### `/modules`
Reusable CMake modules:
- `FindCustomLib.cmake` - Example find module
- `Utils.cmake` - Utility functions
- `PackageBuilder.cmake` - Packaging helpers
- `CodeCoverage.cmake` - Coverage tools

### `/examples`
Additional examples and patterns:
- `integration_examples/` - CMake + Make integration
- `ci_cd_examples/` - Continuous integration examples
- `cross_compilation/` - Cross-platform build examples

## Getting Started

### Basic Projects
Start with the basic projects to learn fundamental concepts:

```bash
# Navigate to a basic project
cd training_projects/basic/hello_world

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make

# Run the executable
./hello
```

### Intermediate Projects
Once comfortable with basics, move to intermediate projects:

```bash
cd training_projects/intermediate/calculator
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make
./calculator
make test  # Run tests
```

### Advanced Projects
For advanced concepts and real-world scenarios:

```bash
cd training_projects/advanced/plugin_architecture
mkdir build && cd build
cmake ..
make
./plugin_demo
```

## Learning Path

### Beginner Track
1. `basic/hello_world` - Basic CMake syntax
2. `basic/library_example` - Creating libraries
3. `basic/multi_file` - Multiple source files
4. `basic/simple_make` - Pure Makefiles

### Intermediate Track
1. `intermediate/calculator` - Multi-file projects with tests
2. `intermediate/logger` - Threading and file I/O
3. Practice with modules in `/modules`

### Advanced Track
1. `advanced/plugin_architecture` - Dynamic loading
2. Explore advanced examples
3. Create your own projects using learned patterns

## Key Concepts Practiced

### CMake Concepts
- `add_executable()` and `add_library()`
- `target_*` commands for proper dependency management
- `find_package()` for external dependencies
- `enable_testing()` and `add_test()` for testing
- `install()` for deployment
- Custom modules and functions

### Make Concepts
- Variable definitions and usage
- Pattern rules with `%`
- Automatic dependency generation (`-MMD -MP`)
- Phony targets
- Conditional execution

### Best Practices Demonstrated
- Modern CMake (target-based approach)
- Proper include directory management
- Cross-platform compatibility
- Testing integration
- Clean separation of interface and implementation

## Troubleshooting

If you encounter issues:

1. Check that all required tools are installed:
   ```bash
   cmake --version
   gcc --version
   make --version
   ```

2. Look at the troubleshooting guide in the main directory

3. Examine the CMake output for error messages

4. For Make issues, use `make --debug=v` for verbose output

## Extending the Training

After completing these projects:
1. Modify existing projects to experiment with new features
2. Create your own projects using these as templates
3. Contribute improvements to these training examples
4. Explore real-world projects to see these concepts in action