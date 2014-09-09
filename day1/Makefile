all: ex3 ex4 ex5

#ex2: ex2 nvcc -o squareArray ex2/squareArray.cu
#	 nvcc -o addArrays ex2/addArrays.cu

ex3: ex3/main.cu aux.cu aux.h Makefile
	nvcc -o ex3/main ex3/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex4: ex4/main.cu aux.cu aux.h Makefile
	nvcc -o ex4/main ex4/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex5: ex5/main.cu aux.cu aux.h Makefile
	nvcc -o ex5/main ex5/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex6: ex6/main.cu aux.cu aux.h Makefile
	nvcc -o ex6/main ex6/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core