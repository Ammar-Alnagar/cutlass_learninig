# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -g
CPPFLAGS = -Iinclude
LDFLAGS =
LDLIBS =

# Directory settings
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DEPENDS = $(OBJECTS:.o=.d)

# Target executable
TARGET = $(BINDIR)/simple_make_app

# Enable automatic dependency generation
CFLAGS += -MMD -MP

# Default target
.PHONY: all
all: $(TARGET)

# Create directories
$(OBJDIR) $(BINDIR):
	@mkdir -p $@

# Compile rule with automatic dependency tracking
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "Compiling $<"
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS) | $(BINDIR)
	@echo "Linking $@"
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Include auto-generated dependencies
-include $(DEPENDS)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(OBJDIR) $(BINDIR)

# Run the application
.PHONY: run
run: $(TARGET)
	@echo "Running application..."
	./$(TARGET)

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all  - Build the application (default)"
	@echo "  clean  - Remove build artifacts"
	@echo "  run  - Build and run the application"
	@echo "  help  - Show this help message"