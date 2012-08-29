
CXXFLAGS=-g2 -ggdb -O0  -msse4 #-DUSE_PAPI_TRACE -funroll-loops -mtune=native 
CC=g++

all: simple.cc special.h
	$(CC) -o simple -DNDEBUG $(CXXFLAGS) simple.cc -L/usr/local/lib #-lpapi

special.h: mask_test.py tpl/all.tpl
	python mask_test.py > special.h

clean:
	$(RM) simple special.h
