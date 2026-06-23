# Makefile for Neuromorphic SNN Project
# Supports: Linux, macOS, HPC clusters

CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
INCLUDES = -Isrc/common

# Detect OS
UNAME_S := $(shell uname -s)

# Sequential (no special flags needed)
sequential: src/sequential/snn_sequential.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o snn_sequential

# OpenMP (add -fopenmp)
openmp: src/openmp/snn_openmp.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fopenmp $< -o snn_openmp

# OpenCL (link against OpenCL)
opencl: src/opencl/snn_opencl.cpp
ifeq ($(UNAME_S),Darwin)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -framework OpenCL $< -o snn_opencl
else
	$(CXX) $(CXXFLAGS) $(INCLUDES) -lOpenCL $< -o snn_opencl
endif

# Build all versions
all: sequential openmp opencl

# Clean build artifacts
clean:
	rm -f snn_sequential snn_openmp snn_opencl *.exe

# Run quick verification tests
test: all
	@echo "=== Testing Sequential ==="
	./snn_sequential --neurons 1000 --timesteps 100 --density 0.1 --seed 42
	@echo "=== Testing OpenMP ==="
	./snn_openmp --neurons 1000 --timesteps 100 --density 0.1 --seed 42
	@echo "=== Testing OpenCL ==="
	./snn_opencl --neurons 1000 --timesteps 100 --density 0.1 --seed 42

.PHONY: all clean test sequential openmp opencl
