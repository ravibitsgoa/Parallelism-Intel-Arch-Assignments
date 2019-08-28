CXX=mpiicpc
CXXFLAGS=-c -qopenmp -qopt-report=5 -xMIC-AVX512
LDFLAGS=-qopenmp

OBJECTS=main.o library.o worker.o

integral: $(OBJECTS)
	$(CXX) $(LDFLAGS) -o integral $(OBJECTS)

all:	integral

run:	all
	./integral

queue:	all
	echo 'cd $$PBS_O_WORKDIR ; mpirun -machinefile $$PBS_NODEFILE ./integral' | qsub -l nodes=4:flat -N numintegr

clean:
	rm -f *.optrpt *.o integral *~ numintegr.*
