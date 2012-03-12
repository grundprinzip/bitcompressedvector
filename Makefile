SHELL = /bin/bash
BUILD_DIR=build

CXXFLAGS= -g2  -mtune=native -mssse3 -msse4.1

all: gen
	mkdir -p $(BUILD_DIR)
	g++ -o $(BUILD_DIR)/main main.cpp $(CXXFLAGS)
	g++ -O3 -o $(BUILD_DIR)/main_opt main.cpp -g2 -DNDEBUG $(CXXFLAGS)

gen:
	cat mask_tpl.h > mask.h
	python generate.py >> mask.h

profile:
	g++ -O3 -o $(BUILD_DIR)/main_opt main.cpp -g2 -DNDEBUG -lprofiler
	CPUPROFILE_FREQUENCY=1000 CPUPROFILE=/tmp/bcv.prof ./$(BUILD_DIR)/main_opt 100000000
	pprof --pdf ./$(BUILD_DIR)/main_opt /tmp/bcv.prof > bcv.pdf

papi:
	g++ -O3 -o $(BUILD_DIR)/main_opt main.cpp -g2 -DNDEBUG -lpapi -DUSE_PAPI_TRACE -lpthread

test:
	./$(BUILD_DIR)/main_opt 1000000
	./$(BUILD_DIR)/main_opt 10000000
	./$(BUILD_DIR)/main_opt 100000000

release: gen
	$(RM) -Rf pkg/bcv
	mkdir -p pkg/bcv
	cp bcv_defines.h pkg/bcv
	cp mask.h pkg/bcv
	cp bcv.h pkg/bcv
	tar -C pkg -zcvf bcv.tgz bcv


clean:
	$(RM) -Rf pkg bcv.tgz $(BUILD_DIR)
