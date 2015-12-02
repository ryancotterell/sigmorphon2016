all: libperceptron.so libalign.so

libperceptron.so: perceptron.c
	gcc -O3 -Wall -Wextra -shared perceptron.c -o libperceptron.so

libalign.so: align.c
	gcc -O3 -Wall -Wextra -shared align.c -o libalign.so

clean:
	/bin/rm libperceptron.so libalign.so *.pyc
