# environment
SM := 35

GCC := g++
NVCC := nvcc

# Remove function
RM = rm -f
 
# Specify opencv Installation
#opencvLocation = /usr/local/opencv
opencvLIB= `pkg-config --libs opencv`
opencvINC= `pkg-config --cflags opencv`

# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart

NVCCFLAGS := -O3
GccFLAGS = -fopenmp -O3 

# The build target executable:
TARGET  = heq
TARGETS = $(TARGET)

all: build

debug: GccFLAGS += -DDEBUG -g -Wall
debug: NVCCFLAGS += -g -G
debug: all

build: $(TARGETS)

$(TARGET): src/dlink.o src/main.o src/$(TARGET).o
	$(NVCC) $(NVCCFLAGS) $(opencvINC) $^ -o $@ $(GENCODE_FLAGS) $(opencvLIB) -link 

src/dlink.o: src/$(TARGET).o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

src/main.o: src/main.cpp
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) -c $< -o $@
	
src/$(TARGET).o: src/$(TARGET).cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 

%: src/%.cpp
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) $< -o $@ $(OPENCV_LINK)
	
clean:
	$(RM) $(TARGETS) src/*.o *.o *.tar* *.core* *out*.jpg *input*.jpg
    
rmlogs:
	$(RM) run_exec.log exec_proj.*.out exec_heq.*.out
    