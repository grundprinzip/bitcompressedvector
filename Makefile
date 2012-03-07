

all: gen
	g++ -o main main.cpp -g2
	g++ -S -O3 main.cpp -g2 -DNDEBUG
	g++ -O3 -o main_opt main.cpp -g2 -DNDEBUG

gen:
	cat bcv_tpl.h > bcv.h
	python generate.py >> bcv.h

clear:
	$(RM) main