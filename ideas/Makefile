
CXXFLAGS=-g -O3 -funroll-loops -msse4 -mtune=native -DUSE_PAPI_TRACE
CC=g++

all: simple.cc special.h
	$(CC) -o simple -DNDEBUG $(CXXFLAGS) simple.cc -L/usr/local/lib -lpapi

special.h: mask_test.py tpl/all.tpl
	python mask_test.py > special.h

clean:
	$(RM) simple special.h
