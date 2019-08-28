CXX=icpc
CXXFLAGS=-xMIC-AVX512 -qopenmp -mkl -std=c++11
OPTRPT=-qopt-report=5

default : app


worker.o : worker.cc
	${CXX} -c ${OPTRPT} ${CXXFLAGS} -o "$@" "$<"

app : main.cc worker.o
	${CXX} ${OPTRPT} ${CXXFLAGS} -o "$@" "$<" worker.o

clean :
	rm app worker.o *.optrpt