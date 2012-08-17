
CXXFLAGS=-g -O3 -funroll-loops -msse4 -mtune=native 
CC=g++

all: simple.cc special.h
	$(CC) -o simple -DNDEBUG $(CXXFLAGS) simple.cc

special.h: mask_test.py tpl/all.tpl
	python mask_test.py > special.h

clean:
	$(RM) simple special.h
