#!/bin/bash

# Setup verification script for CMake and Make learning guide

echo "==========================================="
echo "CMake and Make Learning Guide - Setup Check"
echo "==========================================="

# Check for required tools
echo
echo "Checking for required tools..."

REQUIRED_TOOLS=("cmake" "gcc" "make")
MISSING_TOOLS=()

for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        MISSING_TOOLS+=("$tool")
        echo "❌ $tool: NOT FOUND"
    else
        version=$($tool --version | head -n1)
        echo "✅ $tool: $version"
    fi
done

# Check for recommended tools
echo
echo "Checking for recommended tools..."

RECOMMENDED_TOOLS=("ninja" "doxygen" "clang-format")
for tool in "${RECOMMENDED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        echo "⚠️  $tool: NOT FOUND (recommended but optional)"
    else
        version=$($tool --version | head -n1 2>&1)
        echo "✅ $tool: $version"
    fi
done

# Summary
echo
if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo "❌ Some required tools are missing:"
    for tool in "${MISSING_TOOLS[@]}"; do
        echo "   - $tool"
    done
    echo
    echo "Please install the missing tools before proceeding."
    echo "On Ubuntu/Debian: sudo apt-get install build-essential cmake"
    echo "On macOS: brew install cmake"
    exit 1
else
    echo "✅ All required tools are available!"
fi

# Test basic functionality
echo
echo "Testing basic functionality..."

# Create a temporary test directory
TEST_DIR=$(mktemp -d)
pushd "$TEST_DIR" > /dev/null

# Create a simple CMake test
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(TestSetup)
set(CMAKE_CXX_STANDARD 17)
add_executable(hello hello.cpp)
EOF

cat > hello.cpp << 'EOF'
#include <iostream>
int main() {
    std::cout << "Hello, Setup Test!" << std::endl;
    return 0;
}
EOF

# Try to build the test
if cmake . && make 2>/dev/null; then
    echo "✅ CMake configuration and build: SUCCESS"
    ./hello
else
    echo "❌ CMake configuration or build: FAILED"
    popd > /dev/null
    rm -rf "$TEST_DIR"
    exit 1
fi

popd > /dev/null
rm -rf "$TEST_DIR"

echo
echo "==========================================="
echo "Setup verification: COMPLETE"
echo "You're ready to start the CMake and Make tutorial!"
echo "==========================================="
echo
echo "Next steps:"
echo "1. Read the README.md for an overview"
echo "2. Start with hands_on_tutorial.md"
echo "3. Work through the exercises.md"
echo