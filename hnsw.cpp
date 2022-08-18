#include <string>
#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <time.h>
#include <bits/stdc++.h>
#include <string.h>
#include <iostream>
#include <fstream>

using std::cout;
using std::endl;

//module load compiler/gcc/9.1/openmpi/4.0.2

int get_memory_usage_kb(long* vmpeak_kb, long* vmsize_kb)
{
    /* Get the the current process' status file from the proc filesystem */
    FILE* procfile = fopen("/proc/self/status", "r");

    long to_read = 8192;
    char buffer[to_read];
    int read = fread(buffer, sizeof(char), to_read, procfile);
    fclose(procfile);

    short found_vmpeak = 0;
    short found_vmsize = 0;
    char* search_result;

    /* Look through proc status contents line by line */
    char delims[] = "\n";
    char* line = strtok(buffer, delims);

    while (line != NULL && (found_vmpeak == 0 || found_vmsize == 0) )
    {
        search_result = strstr(line, "VmPeak:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmpeak_kb);
            found_vmpeak = 1;
        }

        search_result = strstr(line, "VmSize:");
        if (search_result != NULL)
        {
            sscanf(line, "%*s %ld", vmsize_kb);
            found_vmsize = 1;
        }

        line = strtok(NULL, delims);
    }

    return (found_vmpeak == 1 && found_vmsize == 1) ? 0 : 1;
}

int get_cluster_memory_usage_kb(long* vmpeak_per_process, long* vmsize_per_process, int root, int np)
{
    long vmpeak_kb;
    long vmsize_kb;
    int ret_code = get_memory_usage_kb(&vmpeak_kb, &vmsize_kb);

    if (ret_code != 0)
    {
        printf("Could not gather memory usage!\n");
        return ret_code;
    }

    MPI_Gather(&vmpeak_kb, 1, MPI_UNSIGNED_LONG, 
        vmpeak_per_process, 1, MPI_UNSIGNED_LONG, 
        root, MPI_COMM_WORLD);

    MPI_Gather(&vmsize_kb, 1, MPI_UNSIGNED_LONG, 
        vmsize_per_process, 1, MPI_UNSIGNED_LONG, 
        root, MPI_COMM_WORLD);

    return 0;
}

int get_global_memory_usage_kb(long* global_vmpeak, long* global_vmsize, int np)
{
    long vmpeak_per_process[np];
    long vmsize_per_process[np];
    int ret_code = get_cluster_memory_usage_kb(vmpeak_per_process, vmsize_per_process, 0, np);

    if (ret_code != 0)
    {
        return ret_code;
    }

    *global_vmpeak = 0;
    *global_vmsize = 0;
    for (int i = 0; i < np; i++)
    {
        *global_vmpeak += vmpeak_per_process[i];
        *global_vmsize += vmsize_per_process[i];
    }

    return 0;
}

void heapify(const int &MAX_OUT, int *heap, float *heap_score, int i=0)
{
    while (2*i+1<MAX_OUT) {
        int smallest = i, ind = 2*i+1;
        if (heap_score[ind] < heap_score[smallest] 
            || (heap_score[ind] == heap_score[smallest] && heap[ind] > heap[smallest])) {
            smallest = ind;
        }
        if (ind+1 < MAX_OUT && (heap_score[ind+1] < heap_score[smallest] 
            || (heap_score[ind+1] == heap_score[smallest] && heap[ind+1] > heap[smallest]))) {
            smallest = ind+1;
        }
        if (smallest == i) {
            return;
        }
        std::swap(heap[i], heap[smallest]);
        std::swap(heap_score[i], heap_score[smallest]);
        i = smallest;
    }
}

void bottom_up_heapify(int i, int *heap, float *heap_score)
{
    while (i) {
        int parent = (i-1)/2;
        if (heap_score[i] < heap_score[parent] 
            || (heap_score[i] == heap_score[parent] && heap[i] > heap[parent])) {
            std::swap(heap[i], heap[parent]);
            std::swap(heap_score[i], heap_score[parent]);
            i = parent;
        }
        else {
            return;
        }
    }
}

void insert_to_heap(int node, float score, int *heap, float *heap_score, const int &k)
{
    if (heap[k-1] == -1) {
    	int i=0;
    	for (; heap[i]!=-1; i++) {}
        heap[i] = node;
        heap_score[i] = score;
        bottom_up_heapify(i, heap, heap_score);
    }
    else if (heap_score[0] < score || (heap_score[0]==score && heap[0] > node)) {
        // insert to heap
        heap[0] = node, heap_score[0] = score; 
        heapify(k, heap, heap_score);
    }
}

void read_array(std::string path, int count, int seek, void *arr1, bool isInt = true)
{
	MPI_File fh_input;
    MPI_File_open(MPI_COMM_WORLD, path.c_str(), MPI_MODE_RDWR, MPI_INFO_NULL, &fh_input);
    // if (seek >= (1<<29)) {
    // 	int tmp = seek/4;
    // 	for (int i=0; i<4; i++)
    // 		MPI_File_set_view(fh_input, tmp*4, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
    // 	MPI_File_set_view(fh_input, (seek-4*(tmp/4))*4, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
    // }
    // else
    
   	MPI_File_seek(fh_input, seek*1ll*4, MPI_SEEK_SET);	

    MPI_Status status;
    if (isInt) {
    	int *arr = (int *)arr1;
    	if (count >= (1<<29)) {
    		MPI_File_read(fh_input, arr, count/3, MPI_INT, &status);
    		MPI_File_read(fh_input, arr+count/3, count/3, MPI_INT, &status);
    		MPI_File_read(fh_input, arr+2*(count/3), count - 2*(count/3), MPI_INT, &status);
    	}
    	else 
    		MPI_File_read(fh_input, arr, count, MPI_INT, &status);
    }
    else {
    	float *arr = (float *)arr1;
    	if (count >= (1<<29)) {
    		MPI_File_read(fh_input, arr, count/3, MPI_FLOAT, &status);
    		MPI_File_read(fh_input, arr+count/3, count/3, MPI_FLOAT, &status);
    		MPI_File_read(fh_input, arr+2*(count/3), count - 2*(count/3), MPI_FLOAT, &status);
    	}
    	else
    		MPI_File_read(fh_input, arr, count, MPI_FLOAT, &status);
    	// std::cout << *arr << "hahahahah" << *(arr+1) << std::endl;
    }
    // std::cout << "ha" << seek << std::endl;
    MPI_File_close(&fh_input);
}

void read_config(std::string out_path, int &num_nodes, int &index_size, int &max_level, int &ep, int &D, int &num_query)
{
	int arr[6];
	read_array(out_path+"/config.dat", 6, 0, arr);
	num_nodes = arr[0], index_size = arr[1], max_level = arr[2], ep = arr[3], D = arr[4], num_query = arr[5];
}

// taken from https://stackoverflow.com/questions/30404099/right-way-to-compute-cosine-similarity-between-two-arrays
float cosine_similarity(float *A, float *B, unsigned int Vector_Length)
{
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(unsigned int i = 0u; i < Vector_Length; ++i) {
        dot += A[i] * B[i] ;
        denom_a += A[i] * A[i] ;
        denom_b += B[i] * B[i] ;
    }
    return std::abs(dot / (sqrt(denom_a) * sqrt(denom_b)));
}

inline int where_is_node(int &u, int &size, int &num_nodes)
{
	int loc = u/(num_nodes/size);
	if (loc == size)
		loc--;
	return loc;
}

void compute_cosine(float *recv_buf, float *send_buf, float *vect, int node_block, int k, int D, int tid, int start)
{
	float *q = recv_buf + tid*(D + node_block);
	int *nodes = (int *)q + D;

	for (int i=0; i<node_block; i++) {
		if (*nodes == -1) {
			return;
		}
		/*int x = 2, y = 1305265;
		std::cout << *nodes << " " << where_is_node(*nodes, x, y) << " " << start << std::endl;*/
		float score = cosine_similarity(q, vect + ((*nodes) - start)*D, D);
		// std::cout << *nodes << "ha" << score << " " << i <<  std::endl;
		insert_to_heap(*nodes, score, (int *)send_buf+tid*2*k, send_buf+tid*2*k + k, k);
		++nodes;
	}
}

void QueryHNSW(float *res, int *heap, float *heap_score, int *indptr, int *index, int *level_offset, 
	int *vect_loc, float *upper_level_vect, int ep, int max_level, int D, int rank, int tid, 
	int size, int n_threads, int k, int num_nodes, int node_block)
{
	float *q = res+(rank*n_threads + tid)*(D + node_block);
	int *res1 = (int *)res;
	/*for (int i=0; i<6; i++)
		std::cout << *(q+i) << " ";
	std::cout << std::endl;*/
	int *visited = new int[num_nodes];
	// int *tmp = new int[size];
	// memset(tmp, 0, size*sizeof(int));
	int tmp[size] = {0};
	memset(visited, 0, num_nodes*sizeof(int));
	std::queue<int> candidates;
	candidates.push(ep);
	visited[ep] = 1;
	if (max_level) {
		float score = cosine_similarity(q, upper_level_vect + vect_loc[ep]*D, D);
		insert_to_heap(ep, score, heap, heap_score, k);
	} 
	else {
		int proc = where_is_node(ep, size, num_nodes);
		// memcpy(res + (proc*n_threads + tid)*(D + node_block) + D + tmp[proc]++, &ep, sizeof(float));
		res1[(proc*n_threads + tid)*(D + node_block) + D + tmp[proc]++] = ep;
	}

	for (int i=max_level; i>=0; i--) {
		while (!candidates.empty()) {
			int u = candidates.front();
			candidates.pop();
			for (int j=level_offset[i]; j<level_offset[i+1]; j++) {
				int px = index[j + indptr[u]];
				if (px == -1 || visited[px]) {
					continue;
				}
				visited[px] = true;
				if (i) {
					float score = cosine_similarity(q, upper_level_vect + vect_loc[px]*D, D);
					insert_to_heap(px, score, heap, heap_score, k);
				}
				else {
					int proc = where_is_node(px, size, num_nodes);
					// memcpy(res + (proc*n_threads + tid)*(D + node_block) + D + tmp[proc]++, &px, sizeof(float));
					res1[(proc*n_threads + tid)*(D + node_block) + D + tmp[proc]++] = px;
				}
				candidates.push(px);
			}
		}
		for (int j=0; j<k; j++) {
			// std::cout << heap[j] << " ";
			if (heap[j] != -1) {
				candidates.push(heap[j]);
			}
		}
		// std::cout << std::endl;
	}
	// std::cout << tmp[0] << "hahfdhdghdgfdg" << std::endl;
	// delete[] tmp;
	delete[] visited;
}

void merge_heaps(float *global_heap, int *heap, float *score, int tid, int size, int k, int n_threads)
{
	/*for (int i=0; i<k; i++) 
		std::cout << heap[i] << "haha" << score[i] << std::endl;*/
	for (int i=0; i<size; i++) {
		for (int j=0; j<k; j++) {
			// std::cout << i << " " <<  *((int *)global_heap + i*n_threads*k*2 + tid*k + j) << " " << global_heap[i*n_threads*k*2 + tid*k + k +j] << std::endl;
			insert_to_heap(*((int *)global_heap + i*n_threads*k*2 + tid*k*2 + j), global_heap[i*n_threads*k*2 + tid*k*2 + k +j], heap, score, k);
		}
	}
}

void heapsort(const int &MAX_OUT, int *heap, float *score)
{
    for (int i=0; i<MAX_OUT; i++) {
        std::swap(heap[MAX_OUT-i-1], heap[0]);
        std::swap(score[MAX_OUT-i-1], score[0]);
        heapify(MAX_OUT-i-1, heap, score);
    }
    // std::reverse(heap, heap+MAX_OUT);
}

int main(int argc, char const *argv[])
{
	auto begin = std::chrono::high_resolution_clock::now();	
	std::string out_path = argv[1];
	int k = std::stoi(argv[2]);
	std::string user_path = argv[3];
	std::string user_output_path = argv[4];

	int rank, size, num_nodes, index_size, max_level, ep, D, num_query;

	int provided;
	MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // std::cout << rank << " " << size << std::endl;
	read_config(out_path, num_nodes, index_size, max_level, ep, D, num_query);
	// std::cout << num_nodes << " " << index_size << " " << max_level << " " << ep << " " << D << " " << num_query << std::endl;
	max_level--;

	int *level = new int[num_nodes];
	int *index = new int[index_size];
	int *indptr = new int[num_nodes+1];
	int *vect_loc = new int[num_nodes];
	int level_offset[max_level+2];
	int node_offset[size+1], query_offset[size+1], upper_level_nodes[size+1] = {0};

	node_offset[size] = num_nodes, query_offset[size] = num_query;
	for (int i=0; i<size; i++) {
		node_offset[i] = (num_nodes/size)*i;
		query_offset[i] = (num_query/size)*i;
		// std::cout << rank << " " << node_offset[i] << " " << query_offset[i] << std::endl;
	}

	float *vect = new float[(node_offset[rank+1] - node_offset[rank])*D];
	float *user = new float[(query_offset[rank+1] - query_offset[rank])*D];
	// int *top_k = new int[(query_offset[rank+1] - query_offset[rank])*k]; (int, float)???

	read_array(out_path+"/level.dat", num_nodes, 0, (void *)level);
	read_array(out_path+"/index.dat", index_size, 0, (void *)index);
	read_array(out_path+"/indptr.dat", num_nodes+1, 0, (void *)indptr);
	read_array(out_path+"/level_offset.dat", max_level+2, 0, (void *)level_offset);
	// std::cout << "here 1 " << node_offset[rank+1] - node_offset[rank] << " " << query_offset[rank+1] - query_offset[rank] << std::endl;
	read_array(out_path+"/vect.dat", (node_offset[rank+1] - node_offset[rank])*D, node_offset[rank]*D, (void *)vect, false);
	// ? use user.txt given in command line argument????
	// std::cout << "haha 2" << std::endl;
	read_array(user_path, (query_offset[rank+1] - query_offset[rank])*D, query_offset[rank]*D, (void *)user, false);
	// std::cout << "haha 1" << std::endl;

	int cur=0, cur1=0;
	for (int i=0; i<num_nodes; i++) {
		level[i]--;
		if (level[i])
			vect_loc[i] = cur++, upper_level_nodes[i/(num_nodes/size)]++;
		else
			vect_loc[i] = -1;
	}
	upper_level_nodes[size-1] += upper_level_nodes[size];

	float *local_upper_level_vect = new float[upper_level_nodes[rank]*D];
	float *upper_level_vect = new float[cur*D];

	for (int i=0; i<node_offset[rank+1]-node_offset[rank]; i++) {
		if (vect_loc[i+node_offset[rank]] != -1) {
			memcpy(local_upper_level_vect+cur1*D, vect+i*D, D*sizeof(float));
			cur1++;
		}
	}
	assert(cur1 == upper_level_nodes[rank]);

	int displs[size] = {0};
	for (int i=1; i<size; i++)
		displs[i] = displs[i-1] + upper_level_nodes[i-1]*D;
	/*for (int i=0; i<size; i++) {
		std::cout << rank << " " << i << " " << upper_level_nodes[i] << " " << displs[i] << std::endl;
	}*/
	int send_count[size] = {0};
	for (int i=0; i<size; i++)
		send_count[i] = upper_level_nodes[i]*D;
	MPI_Allgatherv(local_upper_level_vect, upper_level_nodes[rank]*D, MPI_FLOAT, upper_level_vect, send_count, displs, MPI_FLOAT, MPI_COMM_WORLD);
	// std::cout << "here 2" << std::endl;

	int n_threads, longest_block, longest_loop_length, node_block, *recomm;
	float *res, *recv_buf, *send_buf, *global_heap, *recomm_score;

	n_threads = omp_get_max_threads();
	longest_block = query_offset[size] - query_offset[size-1],
	longest_loop_length = longest_block - (longest_block/n_threads)*(n_threads-1),
	node_block = node_offset[size] - node_offset[size-1];

	res = new float[size*n_threads*(D + node_block)];
	recv_buf = new float[n_threads*(D + node_block)];
	send_buf = new float[n_threads*k*2];
	global_heap = new float[size*n_threads*k*2];
	recomm = new int[(query_offset[rank+1] - query_offset[rank])*k];
	recomm_score = new float[(query_offset[rank+1] - query_offset[rank])*k];

	memset(recomm, -1, (query_offset[rank+1] - query_offset[rank])*k*sizeof(int));

	std::cout << "beginning of the end: " << n_threads << std::endl;

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		
		int block = query_offset[rank+1] - query_offset[rank],
			loop_length = block/n_threads, start = loop_length*tid;
		if (tid == n_threads-1)
			loop_length = block - (block/loop_length)*(n_threads-1);

		for (int i=0; i<longest_loop_length; i++) {
			// memset(res+tid*size*(D + node_block), -1, size*(D + node_block)*sizeof(float));
			for (int j=0; j<size; j++)
				memset(res+(j*n_threads + tid)*(D + node_block), -1, (D + node_block)*sizeof(float));
			

			#pragma omp barrier

			if (i<loop_length) {
				// compute result
				for (int j=0; j<size; j++)
					memcpy(res+(j*n_threads + tid)*(D + node_block), user+(start+i)*D, D*sizeof(float));
				QueryHNSW(res, recomm + (start+i)*k, recomm_score + (start+i)*k, indptr, index, 
					level_offset, vect_loc, upper_level_vect, ep, max_level, D, rank, tid, size,
					n_threads, k, num_nodes, node_block);

				/*for (int j=0; j<k; j++) {
					std::cout << recomm[(start+i)*k + j] << "haha" << recomm_score[(start+i)*k + j] << " " << rank << " " << tid << std::endl;
				}*/
				// std::cout << std::endl;
			}

			#pragma omp barrier

			for (int j=0; j<size; j++) {
				if (tid == 0) {	
					MPI_Scatter(res, n_threads*(D+node_block), MPI_FLOAT, recv_buf, n_threads*(D+node_block), MPI_FLOAT, j, MPI_COMM_WORLD);
				}
				#pragma omp barrier
				
				memset(send_buf + tid*k*2, -1, k*2*sizeof(float));
				compute_cosine(recv_buf, send_buf, vect, node_block, k, D, tid, node_offset[rank]);

				/*for (int l=0; l<k; l++) {
					std::cout << *((int*)send_buf+tid*k*2 + l) << " " << send_buf[l+tid*k*2+k] << " " << rank << " " << j << std::endl;
				}*/

				#pragma omp barrier
				if (tid == 0) {
					MPI_Gather(send_buf, n_threads*k*2, MPI_FLOAT, global_heap, n_threads*k*2, MPI_FLOAT, j, MPI_COMM_WORLD);
				}
				#pragma omp barrier
				// merge all heaps and computer final result
	
				if (rank == j)
					merge_heaps(global_heap, recomm + (start+i)*k, recomm_score + (start+i)*k, tid, size, k, n_threads);

				#pragma omp barrier
			}

			heapsort(k, recomm + (start+i)*k, recomm_score + (start+i)*k);
			
			/*std::cout << query_offset[rank] + (start+i) << std::endl;
			for (int j=0; j<k; j++) {
				std::cout << recomm[(start+i)*k + j] << " " << recomm_score[(start+i)*k + j] << endl;
			}
			std::cout << std::endl;
			
			if (i==10)
				break;*/
		}
	}

	if (rank == 0) {
		int *all_recomm = new int[num_query*k];
		int recv_count[size] = {0};
		for (int i=0; i<size; i++)
			displs[i] = query_offset[i]*k, recv_count[i] = (query_offset[i+1]-query_offset[i])*k;
		MPI_Gatherv(recomm, (query_offset[rank+1]-query_offset[rank])*k, MPI_FLOAT, all_recomm, recv_count, displs, MPI_INT, 0, MPI_COMM_WORLD);
		std::ofstream myfile;
		myfile.open(user_output_path);
		for (int i=0; i<num_query; i++) {
			for (int j=0; j<k; j++) {
				myfile << all_recomm[i*k+j] << " ";
			}
			myfile << "\n";
		}
		myfile.flush();
		myfile.close();
	}
	else {
		MPI_Gatherv(recomm, (query_offset[rank+1]-query_offset[rank])*k, MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}


	long vmpeak_per_process[size];
	long vmsize_per_process[size];
	get_cluster_memory_usage_kb(vmpeak_per_process, vmsize_per_process, 0, size);

	if (rank == 0)
	{
	    for (int i = 0; i < size; i++)
	    {
	        printf("Process %03d: VmPeak = %6ld KB\n", 
	            i, vmpeak_per_process[i]);
	    }
	}

	long global_vmpeak, global_vmsize;
	get_global_memory_usage_kb(&global_vmpeak, &global_vmsize, size);
	if (rank == 0)
	{
	    printf("Global memory usage: VmPeak = %6ld KB\n", 
	        global_vmpeak);
	}


	MPI_Finalize();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = (1e-6 * (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)).count());
    if (rank == 0) {
        std::cout << "Total time taken: " << duration << std::endl;
        std::cout << "Total time taken per user: " << duration/num_query << std::endl;
    }

    
	return 0;
}