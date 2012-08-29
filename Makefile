SHELL = /bin/bash
BUILD_DIR=build

CC=g++
CXXFLAGS= -mtune=native -mssse3 -msse4.1 -m64

FILES=main.cc test.cc

all: 
	mkdir -p $(BUILD_DIR)
	$(CC) -o $(BUILD_DIR)/main $(FILES) $(CXXFLAGS) -g2 -O0 -ggdb
	$(CC) -o $(BUILD_DIR)/main_opt $(FILES) -DNDEBUG $(CXXFLAGS) -funroll-loops -O3 -g

profile:
	$(CC) -O3 -o $(BUILD_DIR)/main_opt $(FILES) -g2 -DNDEBUG -lprofiler $(CXXFLAGS)
	CPUPROFILE_FREQUENCY=1000 CPUPROFILE=/tmp/bcv.prof ./$(BUILD_DIR)/main_opt 100000000
	pprof --pdf ./$(BUILD_DIR)/main_opt /tmp/bcv.prof > bcv.pdf

papi:
	g++ -O3 -o $(BUILD_DIR)/main_opt $(FILES) -g2 -DNDEBUG -lpapi -DUSE_PAPI_TRACE -lpthread

test: all
	./$(BUILD_DIR)/main 1000
	#./$(BUILD_DIR)/main 1000000
	#./$(BUILD_DIR)/main 10000000

release: gen
	$(RM) -Rf pkg/bcv
	mkdir -p pkg/bcv
	cp bcv_defines.h pkg/bcv
	cp mask.h pkg/bcv
	cp bcv.h pkg/bcv
	tar -C pkg -zcvf bcv.tgz bcv


clean:
	$(RM) -Rf pkg bcv.tgz $(BUILD_DIR)
