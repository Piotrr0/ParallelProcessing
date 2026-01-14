#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>

void print_process_details(int rank, int size) {
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_length;
    MPI_Get_processor_name(processor_name, &name_length);

    std::cout << rank << ": Hello! Processor: " << processor_name << ", World size: " << size << std::endl;
}

void perform_ring_communication(int rank, int size) {
    int token;
    int dest = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

    if (rank == 0) {
        token = 100;
        MPI_Send(&token, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        MPI_Recv(&token, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << rank << ": Ring completed. Token received back: " << token << std::endl;
    } else {
        MPI_Recv(&token, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        token += 1;
        MPI_Send(&token, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        std::cout << rank << ": Received and passed token: " << token << std::endl;
    }
}

void perform_broadcast(int rank) {
    int data;
    if (rank == 0) {
        data = 777;
        std::cout << rank << ": Root broadcasting value: " << data << std::endl;
    }

    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        std::cout << rank << ": Received via broadcast: " << data << std::endl;
    }
}

void compute_parallel_sum(int rank, int size) {
    const int TOTAL_SIZE = 10000;
    int items_per_proc = TOTAL_SIZE / size;
    
    std::vector<int> data;
    long long expected_sum = 0;

    if (rank == 0) {
        data.resize(TOTAL_SIZE, 1);
        expected_sum = (long long)TOTAL_SIZE;
    }

    std::vector<int> local_chunk(items_per_proc);
    MPI_Scatter(data.data(), items_per_proc, MPI_INT, 
                local_chunk.data(), items_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    long long local_sum = std::accumulate(local_chunk.begin(), local_chunk.end(), 0LL);
    
    std::vector<long long> partial_sums;
    if (rank == 0) partial_sums.resize(size);

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG, 
               partial_sums.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long total_sum = std::accumulate(partial_sums.begin(), partial_sums.end(), 0LL);
        std::cout << rank << ": Scatter/Gather result: " << total_sum 
                  << " (Expected: " << expected_sum << ")" << std::endl;
        
        if (total_sum == expected_sum) {
            std::cout << rank << ": Verification successful!" << std::endl;
        } else {
            std::cout << rank << ": Verification failed!" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Process Details \n";
    MPI_Barrier(MPI_COMM_WORLD);
    print_process_details(rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Ring Communication \n";
    MPI_Barrier(MPI_COMM_WORLD);
    perform_ring_communication(rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Broadcast \n";
    MPI_Barrier(MPI_COMM_WORLD);
    perform_broadcast(rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Parallel Sum \n";
    MPI_Barrier(MPI_COMM_WORLD);
    compute_parallel_sum(rank, size);

    MPI_Finalize();
    return 0;
}