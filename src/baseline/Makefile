all: libperceptron.so libalign.so

libperceptron.so: perceptron.c
	gcc -O3 -Wall -Wextra -shared -fPIC perceptron.c -o libperceptron.so

libalign.so: align.c
	gcc -O3 -Wall -Wextra -shared -fPIC align.c -o libalign.so

clean:
	/bin/rm libperceptron.so libalign.so *.pyc
