# Compiler
CXX := g++

# Determine the number of CPU cores for parallel compilation
NUM_CORES := $(shell nproc)

# Default target
.DEFAULT_GOAL := parallel

# Compiler flags
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -Wno-deprecated-declarations -g -rdynamic -fopenmp \
            -m64 -fPIC -fno-strict-aliasing -fexceptions -DIL_STD

# Catch2 paths (adjust if necessary)
CATCH2_INC := /usr/local/include
CATCH2_LIB := /usr/local/lib

EIGEN_INC := /usr/include/eigen3

# Include flags
INCLUDES := -I$(BOOST_INC) -I$(CATCH2_INC) -I$(EIGEN_INC) -Isrc

# Library flags
LDFLAGS := -L$(BOOST_LIB) -L$(CATCH2_LIB)  -lCatch2Main -lCatch2
LDFLAGS +=  -lboost_program_options -fopenmp
LDFLAGS += -lm -lpthread -ldl

# Source files
COMMON_SRC := $(wildcard src/transformer/*.cpp) \
 		      $(wildcard src/types/*.cpp) \
              src/utils.cpp src/argument_parser.cpp src/logger.cpp \
			  src/vocab.cpp
               

SRCS := src/main.cpp $(COMMON_SRC)
TEST_SRCS :=  $(wildcard tests/*.cpp) $(COMMON_SRC)

# Object files
OBJ_DIR := obj
OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TEST_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(TEST_SRCS))

# Executable names
TARGET := tform
TEST_TARGET := tform_test
DEBUG_TARGET := tform__debug

# Test arguments
TEST_ARGS ?=

# Release build settings
RELEASE_CXXFLAGS := $(CXXFLAGS) -O3 -DNDEBUG

# Debug build settings
DEBUG_OBJ_DIR := obj_debug
DEBUG_OBJS := $(patsubst %.cpp,$(DEBUG_OBJ_DIR)/%.o,$(SRCS))
DEBUG_CXXFLAGS := $(CXXFLAGS) -O0 -DDEBUG -fno-omit-frame-pointer -fsanitize=address

# Build targets
build: $(TARGET) $(TEST_TARGET)

# Main executable rule
$(TARGET): $(OBJS)
	$(CXX) $(RELEASE_CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Test executable rule
$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $(RELEASE_CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(RELEASE_CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Debug rules
debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(CXX) $(DEBUG_CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) -fsanitize=address

$(DEBUG_OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(DEBUG_CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Test target (build and run tests)
test: build
	./$(TEST_TARGET) -s $(TEST_ARGS)

# Clean rule
clean:
	rm -rf $(OBJ_DIR) $(DEBUG_OBJ_DIR) $(TARGET) $(TEST_TARGET) $(DEBUG_TARGET)

# Include dependency files
-include $(OBJS:.o=.d)
-include $(DEBUG_OBJS:.o=.d)
-include $(TEST_OBJS:.o=.d)

# Parallel compilation
parallel:
	@$(MAKE) -j$(NUM_CORES) build

# Sequential build (for comparison or troubleshooting)
sequential: build

# Print paths for debugging
paths:
	@echo "Catch2 Include Path: $(CATCH2_INC)"
	@echo "Catch2 Library Path: $(CATCH2_LIB)"
	@echo "Eigen Include Path: $(EIGEN_INC)"
	@echo "Python Version: $(PYTHON_VERSION)"
	@echo "INCLUDES: $(INCLUDES)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "Number of CPU cores: $(NUM_CORES)"

.PHONY: build debug clean parallel test paths sequential