all: ex3 ex4 ex5 ex6 ex7_bonus

ex2: nvcc -o squareArray ex2/squareArray.cu
	 nvcc -o addArrays ex2/addArrays.cu

ex3: ex3/main.cu aux.cu aux.h Makefile
	nvcc -o ex3/main ex3/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex4: ex4/main.cu aux.cu aux.h Makefile
	nvcc -o ex4/main ex4/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex5: ex5/main.cu aux.cu aux.h Makefile
	nvcc -o ex5/main ex5/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex6: ex6/main.cu aux.cu aux.h Makefile
	nvcc -o ex6/main ex6/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex7: ex7/main.cu aux.cu aux.h Makefile
	nvcc -o ex7/main ex7/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex7_bonus: ex7_bonus/main.cu aux.cu aux.h Makefile
		nvcc -o ex7_bonus/main ex7_bonus/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core -lcublas

ex8: ex8/main.cu aux.cu aux.h Makefile
	nvcc -o ex8/main ex8/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex9: ex9/main.cu aux.cu aux.h Makefile
	nvcc -o ex9/main ex9/main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

ex7_tex: ex7/main_tex.cu aux.cu aux.h Makefile
	nvcc -o ex7/main_tex ex7/main_tex.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

