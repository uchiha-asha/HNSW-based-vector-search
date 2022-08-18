compile:
	g++ -o convert convert.cpp
	mpic++ -fopenmp -g -O2 -std=c++17 -o hnsw hnsw.cpp 

run:
	./HNSWpred.sh /home/cse/btech/cs1190337/col380/A3/converted 5 /home/cse/btech/cs1190337/col380/A3/converted/user.dat output.txt

clean:
	rm convert hnsw