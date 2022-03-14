path = /home/avval/Desktop/2020CS10401_2020CS10330
all:
	g++ -L$(path) -Wall -o yourcode.out main.cpp -laudio -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
remove:
	@rm yourcode.out