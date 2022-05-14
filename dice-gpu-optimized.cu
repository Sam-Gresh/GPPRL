#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <inttypes.h>



__global__ void find_magnitudes(char* clks, int_fast16_t* output){
    int index = threadIdx.x;
    int block_index = blockIdx.x;
    int32_t* clk_ints = (int32_t*) clks;

    int32_t a = clk_ints[index + 128 * block_index];
    int local_sum = 0;
    while(a){
        a = (a & (a - 1));
        local_sum++;
    }

    typedef cub::BlockReduce<int, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __syncthreads();
    int aggregate = BlockReduce(temp_storage).Sum(local_sum);
    if(index == 0){
        output[block_index] = aggregate;
    }
}

__global__ void find_dice_coeff(char* clks1, char* clks2, int_fast16_t num_clks1, int_fast64_t num_clks2, int_fast64_t x_stride, int_fast64_t y_stride, int_fast16_t* mags1, int_fast16_t* mags2, int_fast64_t* output, float threshhold){
    typedef cub::BlockReduce<int, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int32_t* clk_ints1 = (int32_t*) clks1;
    int32_t* clk_ints2 = (int32_t*) clks2;

    for(int_fast64_t i = blockIdx.x; i < num_clks1; i += x_stride){
        for(int_fast64_t j = blockIdx.y; j < num_clks2; j += y_stride){
            int index = threadIdx.x;
            int_fast64_t block_index_x = i;
            int_fast64_t block_index_y = j;
            

            
            int32_t a = clk_ints1[index + 128 * block_index_x];
            int32_t b = clk_ints2[index + 128 * block_index_y];
            int32_t c = a & b;
            int local_sum = 0;
            while(c){
                c = (c & (c - 1));
                local_sum++;
            }

            __syncthreads();
            int dot_prod = BlockReduce(temp_storage).Sum(local_sum);
            if(index == 0){
                if(2 * dot_prod >= threshhold * (mags1[block_index_x] + mags2[block_index_y])){
                    output[block_index_x] = block_index_y + 1;
                }
            }
        }
    }
}


int main(int argc, char* argv[]){

    float threshold;
    if(argc < 2){
        threshold = 0.85;
    }
    else{
        threshold = strtod(argv[1], NULL);
    }

    clock_t begin = clock();
    //Dataset1
    FILE* dataset1 = fopen("./dataset1.bin", "rb");
    fseek(dataset1, 0, SEEK_END);
    int_fast64_t dataset1_size_bytes = ftell(dataset1);
    rewind(dataset1);
    int_fast64_t num_clks_dataset1 = dataset1_size_bytes / 512;
    
    char* clks_dataset1 = (char*) malloc(512 * num_clks_dataset1);
    char* d_clks_dataset1;
    size_t pitch;
    cudaMallocPitch(&d_clks_dataset1, &pitch, 512, num_clks_dataset1);

    int_fast16_t* d_mags1;
    cudaMalloc(&d_mags1, num_clks_dataset1 * sizeof(int_fast16_t));

    fread(clks_dataset1, 1, 512 * num_clks_dataset1, dataset1);
    cudaMemcpy2D(d_clks_dataset1, pitch, clks_dataset1, 512, 512, num_clks_dataset1, cudaMemcpyHostToDevice);
    free(clks_dataset1);

    find_magnitudes<<<num_clks_dataset1, 128>>>(d_clks_dataset1, d_mags1);
    printf("Found Dataset 1 magnitudes...\n");



    //Dataset2
    FILE* dataset2 = fopen("./dataset2.bin", "rb");
    fseek(dataset2, 0, SEEK_END);
    int_fast64_t dataset2_size_bytes = ftell(dataset2);
    rewind(dataset2);
    int_fast64_t num_clks_dataset2 = dataset2_size_bytes / 512;
    
    char* clks_dataset2 = (char*) malloc(512 * num_clks_dataset2);
    char* d_clks_dataset2;
    cudaMallocPitch(&d_clks_dataset2, &pitch, 512, num_clks_dataset2);

    int_fast16_t* d_mags2;
    cudaMalloc(&d_mags2, num_clks_dataset2 * sizeof(int_fast16_t));

    fread(clks_dataset2, 1, 512 * num_clks_dataset2, dataset2);
    cudaMemcpy2D(d_clks_dataset2, pitch, clks_dataset2, 512, 512, num_clks_dataset2, cudaMemcpyHostToDevice);
    free(clks_dataset2);


    find_magnitudes<<<num_clks_dataset2, 128>>>(d_clks_dataset2, d_mags2);
    printf("Found Dataset 2 magnitudes...\n");
    
    //dotproducts
    int_fast64_t* d_output;
    cudaMalloc(&d_output, num_clks_dataset1 * sizeof(int_fast64_t));


    dim3 threads_per_block(128);
    int_fast64_t blocks_x = 128;
    int_fast64_t blocks_y = 128;
    dim3 num_blocks(blocks_x, blocks_y);
    find_dice_coeff<<<num_blocks, threads_per_block>>>(d_clks_dataset1, d_clks_dataset2, num_clks_dataset1, num_clks_dataset2, blocks_x, blocks_y, d_mags1, d_mags2, d_output, threshold);

    int_fast64_t* output = (int_fast64_t*) malloc(num_clks_dataset1 * sizeof(int_fast64_t));
    cudaMemcpy(output, d_output, num_clks_dataset1 * sizeof(int_fast64_t), cudaMemcpyDeviceToHost);
    printf("Finished computation...\n");
    FILE* outfile = fopen("./matches.csv", "w");
    for(int_fast64_t i = 0; i < num_clks_dataset1; i++){
        fprintf(outfile, "%" PRId64 ",%" PRId64 "\n", i, output[i] - 1);
    }
    printf("done.\n");
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("The elapsed time is %f seconds\n", time_spent);
    return 0;
}