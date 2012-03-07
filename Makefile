

all: gen
	g++ -o main main.cpp -g2
	g++ -S -O3 main.cpp -g2 -DNDEBUG
	g++ -O3 -o main_opt main.cpp -g2 -DNDEBUG

gen:
	cat mask_tpl.h > mask.h
	python generate.py >> mask.h

clear:
	$(RM) main