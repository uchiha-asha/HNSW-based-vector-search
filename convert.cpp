// /scratch/cse/phd/anz198717/TA/COL380/A3/to_students


#include <stdio.h>
#include <iostream>
#include <bitset>
#include <fstream>
#include <vector>
#include <climits>
#include <bits/stdc++.h>

using namespace std;

int convert(std::string name, std::string in_path, std::string out_path) {
	std::ifstream fp_input((in_path + "/" + name + ".txt").c_str());

	vector<int> v(istream_iterator<int>(fp_input), {});

	/*int x;
    while(fp_input >> x) {
        v.push_back(x);
    }*/

    fp_input.close();

    if (v.size() == 1) {
    	return v[0];
    }

    int *f = new int[v.size()];
    for (int i=0; i<v.size(); i++)
    	f[i] = v[i];

    char *buf = (char *)f;
    FILE *fp = fopen((out_path + "/" + name + ".dat").c_str(), "w");
    fwrite(buf, 1, v.size()*sizeof(int), fp);
    fclose(fp);

    return v.size();
}

int convert_float(std::string name, std::string in_path, std::string out_path) {
	std::ifstream fp_input((in_path + "/" + name + ".txt").c_str());

	vector<float> v(istream_iterator<float>(fp_input), {});

	/*float x;
    while(fp_input >> x) {
        v.push_back(x);
    }*/

    fp_input.close();

    float *f = new float[v.size()];
    for (int i=0; i<v.size(); i++)
    	f[i] = v[i];

    char *buf = (char *)f;
    FILE *fp = fopen((out_path + "/" + name + ".dat").c_str(), "w");
    fwrite(buf, 1, v.size()*sizeof(int), fp);
    fclose(fp);

    return v.size();
}
  
int main(int argc, char** argv) {
	if(argc != 3) return -1;

	std::string in_path = argv[1];
	std::string out_path = argv[2];

	int arr[6];
	arr[0] = convert("level", in_path, out_path); // num_nodes
	arr[1] = convert("index", in_path, out_path); // index_size
	arr[2] = convert("max_level", in_path, out_path);
	arr[3] = convert("ep", in_path, out_path);
	arr[4] = convert_float("vect", in_path, out_path)/arr[0]; // D
	arr[5] = convert_float("user", in_path, out_path)/arr[4]; // num_query
	convert("level_offset", in_path, out_path);
	convert("indptr", in_path, out_path);

	char *buf = (char *)arr;
	FILE *fp = fopen((out_path + "/config.dat").c_str(), "w");
	fwrite(buf, 1, 6*sizeof(int), fp);
	fclose(fp);
}