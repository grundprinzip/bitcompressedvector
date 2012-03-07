

all: gen
	g++ -o main main.cpp -g2
	g++ -S -O3 main.cpp -g2 -DNDEBUG
	g++ -O3 -o main_opt main.cpp -g2 -DNDEBUG

gen:
	cat mask_tpl.h > mask.h
	python generate.py >> mask.h


test:
	./main_opt 1000000
	./main_opt 10000000
	./main_opt 100000000


clear:
	$(RM) main
