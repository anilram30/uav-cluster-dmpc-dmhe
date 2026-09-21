# Build the C numerical core (reference impl) + C++ coordination demo.
CC      ?= gcc
CXX     ?= g++
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -I../c_api
CXXFLAGS?= -O2 -std=c++17 -Wall -Wextra -I../c_api -I.
LDFLAGS ?= -lm

all: cluster_demo

uav_solver_ref.o: ../c_api/uav_solver_ref.c ../c_api/uav_solver.h
	$(CC) $(CFLAGS) -c $< -o $@

cluster_demo: cluster_demo.cpp agent.hpp uav_solver_ref.o
	$(CXX) $(CXXFLAGS) cluster_demo.cpp uav_solver_ref.o -o $@ $(LDFLAGS)

run: cluster_demo
	./cluster_demo

clean:
	rm -f *.o cluster_demo

.PHONY: all run clean
