# External Makefile for integration example
.PHONY: all clean setup

all:
	@echo "Running external Makefile"
	@echo "This could build legacy components"
	@echo "Or run custom build steps"

clean:
	@echo "Cleaning external build artifacts"
	@rm -f external_artifact.*

setup:
	@echo "Setting up external dependencies"
	@mkdir -p external_output