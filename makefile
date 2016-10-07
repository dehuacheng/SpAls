UNAME := $(shell uname)
 
# alsomp:	main.cpp TensorData.cpp TensorDataSpAls.cpp  CPDecomp.cpp asa007.cpp SpAlsUtils.cpp TensorAls.cpp

ifeq ($(UNAME), Linux)
    CC=g++
    LFLAGS = -fopenmp
    TESTFILE=example/tensorTest.csv
endif
ifeq ($(UNAME), Darwin)
    CC=g++-5
    LFLAGS = -fopenmp -L"/usr/local/Cellar/gcc/5.3.0/lib/gcc/5/" -I"include"
    TESTFILE=example/tensorTest.csv
endif

CFLAGS =  -O2 -std=c++11 -I"include" -fopenmp

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

all: $(OBJ_FILES)
	${CC} $(LFLAGS) -o bin/main $^

obj/%.o: src/%.cpp
	${CC} $(CFLAGS) -c -o $@ $<

clean: 
	rm ${OBJ_FILES}

test:
	./bin/main ${TESTFILE}

