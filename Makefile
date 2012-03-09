

all: gen
	g++ -o main main.cpp -g2
	g++ -S -O3 main.cpp -g2 -DNDEBUG
	g++ -O3 -o main_opt main.cpp -g2 -DNDEBUG

gen:
	cat mask_tpl.h > mask.h
	python generate.py >> mask.h

profile:
	g++ -O3 -o main_opt main.cpp -g2 -DNDEBUG -lprofiler
	CPUPROFILE_FREQUENCY=1000 CPUPROFILE=/tmp/bcv.prof ./main_opt 100000000
	pprof --pdf ./main_opt /tmp/bcv.prof > bcv.pdf


test:
	./main_opt 1000000
	./main_opt 10000000
	./main_opt 100000000


clear:
	$(RM) main
