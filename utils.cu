#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK 256
cudaDeviceProp deviceProp;

// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.
__device__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 1;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src){
    int i = 0;
    while ((dest[i] = src[i]) != '\0') {
        i++;
    }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src){
    while (*dest != '\0') {
        dest++;
    }
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
    size_t input_length = d_strlen((const char *)input);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE - 1; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}

__global__ void sha256_transaction_kernel(BYTE *transactions, int transaction_size, BYTE *hashes, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= n) return;
   apply_sha256(transactions + i * transaction_size, hashes + i * SHA256_HASH_SIZE);
}

__device__ int mini(int a, int b) {
	if(a < b) return a;
	return b;
}

__global__ void combine_hashes_kernel(BYTE *in_buf, BYTE *out_buf, int n) {
	__shared__ BYTE shared_hashes[THREADS_PER_BLOCK * SHA256_HASH_SIZE];
   int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
		d_strcpy((char *)(shared_hashes + threadIdx.x * SHA256_HASH_SIZE), (const char *)(in_buf + i * SHA256_HASH_SIZE));
	__syncthreads();

	int local_n = mini(blockDim.x, n - blockIdx.x * blockDim.x);

	BYTE combined[2 * SHA256_HASH_SIZE];
	int curr_n = local_n;

	while(curr_n > 1) {
		int new_n = (curr_n + 1) / 2;
		if(threadIdx.x < new_n) {
			int left_elem = threadIdx.x * 2;
			int right_elem = (left_elem + 1 < curr_n) ? left_elem + 1 : left_elem;
			d_strcpy((char *)combined, (const char *)(shared_hashes + left_elem * SHA256_HASH_SIZE));
			d_strcat((char *)combined, (const char *)(shared_hashes + right_elem * SHA256_HASH_SIZE));
		}
		__syncthreads();

		if(threadIdx.x < new_n)
			apply_sha256(combined, shared_hashes + threadIdx.x * SHA256_HASH_SIZE);
		__syncthreads();

		curr_n = new_n;
	}

	if(threadIdx.x == 0)
		d_strcpy((char *)out_buf + blockIdx.x * SHA256_HASH_SIZE, (const char *)shared_hashes);
}

__global__ void pair_combine_kernel(BYTE *in_buf, BYTE *out_buf, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int new_n = (n + 1) / 2;
	if (i >= new_n) return;

	int left = i * 2;
	int right = (left + 1 < n) ? left + 1 : left;

	BYTE combined[2 * SHA256_HASH_SIZE];
	d_strcpy((char *)combined, (const char *)(in_buf + left * SHA256_HASH_SIZE));
	d_strcat((char *)combined, (const char *)(in_buf + right * SHA256_HASH_SIZE));
	apply_sha256(combined, out_buf + i * SHA256_HASH_SIZE);
}

// TODO 1: Implement this function in CUDA
void construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]) {
	int pool_size = max_transactions_in_a_block * transaction_size + 2 * max_transactions_in_a_block * SHA256_HASH_SIZE;
	BYTE *gpu_pool;
	cudaMalloc(&gpu_pool, pool_size);
	if(!gpu_pool) {
		fprintf(stderr, "Failed to allocate GPU memory for Merkle root construction.\n");
		exit(1);
	}

	BYTE *gpu_transactions = gpu_pool;
	BYTE *gpu_hashes = gpu_pool + max_transactions_in_a_block * transaction_size;
	BYTE *gpu_hashes_buffer = gpu_pool + max_transactions_in_a_block * transaction_size + max_transactions_in_a_block * SHA256_HASH_SIZE;

	cudaMemcpy(gpu_transactions, transactions, n * transaction_size, cudaMemcpyHostToDevice);

	int blocks_sha256_transaction = n / THREADS_PER_BLOCK;
	if (n % THREADS_PER_BLOCK)
		blocks_sha256_transaction++;

	sha256_transaction_kernel<<<blocks_sha256_transaction, THREADS_PER_BLOCK>>>(gpu_transactions, transaction_size, gpu_hashes, n);

	while (n > 1) {
		if (n <= THREADS_PER_BLOCK) {
			combine_hashes_kernel<<<1, THREADS_PER_BLOCK>>>(gpu_hashes, gpu_hashes_buffer, n);
			BYTE *temp = gpu_hashes;
			gpu_hashes = gpu_hashes_buffer;
			gpu_hashes_buffer = temp;
			n = 1;
		} else {
			int new_n = (n + 1) / 2;
			int blocks = new_n / THREADS_PER_BLOCK;
			if (new_n % THREADS_PER_BLOCK)
				blocks++;
			pair_combine_kernel<<<blocks, THREADS_PER_BLOCK>>>(gpu_hashes, gpu_hashes_buffer, n);
			BYTE *temp = gpu_hashes;
			gpu_hashes = gpu_hashes_buffer;
			gpu_hashes_buffer = temp;
			n = new_n;
		}
	}

	cudaMemcpy(merkle_root, gpu_hashes, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
	cudaFree(gpu_pool);
}

__global__ void find_nonce_kernel(BYTE *block_content, size_t current_length, BYTE *difficulty, uint32_t start_nonce, uint32_t end_nonce, uint32_t *result_nonce, volatile uint32_t *global_found) {
   if (*global_found) return;

   uint32_t i = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= end_nonce) return;

   __shared__ BYTE shared_block[BLOCK_SIZE];
   for (int j = threadIdx.x; j <= (int)current_length; j += blockDim.x)
      shared_block[j] = block_content[j];
   __syncthreads();

   BYTE local_block[BLOCK_SIZE];
   d_strcpy((char *)local_block, (const char *)shared_block);

   char nonce_string[NONCE_SIZE];
   intToString(i, nonce_string);
   d_strcpy((char *)local_block + current_length, (const char *)nonce_string);

   BYTE hash[SHA256_HASH_SIZE];
   apply_sha256(local_block, hash);

   if (compare_hashes(hash, difficulty) <= 0) {
      atomicExch((uint32_t*)global_found, 1);
		atomicExch(result_nonce, i);
   }
}

// TODO 2: Implement this function in CUDA
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
	size_t pool_size = BLOCK_SIZE + SHA256_HASH_SIZE + sizeof(uint32_t) + sizeof(uint32_t);
	BYTE *gpu_pool;
	cudaMalloc(&gpu_pool, pool_size);
	if(!gpu_pool) {
		fprintf(stderr, "Failed to allocate GPU memory for nonce finding.\n");
		exit(1);
	}

	BYTE *gpu_block_content = gpu_pool;
	BYTE *gpu_difficulty = gpu_pool + BLOCK_SIZE;
	uint32_t *gpu_result_nonce = (uint32_t *)(gpu_pool + BLOCK_SIZE + SHA256_HASH_SIZE);
	uint32_t *gpu_found = (uint32_t *)(gpu_pool + BLOCK_SIZE + SHA256_HASH_SIZE + sizeof(uint32_t));

	cudaMemcpy(gpu_block_content, block_content, current_length, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);

	int blocks = deviceProp.multiProcessorCount;

	uint32_t found_nonce = 0;
	uint32_t found_flag = 0;

	for (uint32_t start = 0; start < max_nonce && !found_flag; start += (uint32_t)(THREADS_PER_BLOCK * blocks)) {
		uint32_t end = start + (uint32_t)(THREADS_PER_BLOCK * blocks);
		if (end > max_nonce) end = max_nonce;
		int curr_blocks = (end - start) / THREADS_PER_BLOCK;
		if ((end - start) % THREADS_PER_BLOCK)
			curr_blocks++;

		cudaMemset(gpu_found, 0, sizeof(uint32_t));
		cudaMemset(gpu_result_nonce, 0, sizeof(uint32_t));

		find_nonce_kernel<<<curr_blocks, THREADS_PER_BLOCK>>>(gpu_block_content, current_length, gpu_difficulty, start, end, gpu_result_nonce, gpu_found);
		cudaDeviceSynchronize();

		cudaMemcpy(&found_flag, gpu_found, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (found_flag)
			cudaMemcpy(&found_nonce, gpu_result_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	}

	cudaFree(gpu_pool);

	if (!found_flag)
		return 1;

	*valid_nonce = found_nonce;
	char nonce_string[NONCE_SIZE];
	sprintf(nonce_string, "%u", found_nonce);
	strcpy((char *)(block_content + current_length), (const char *)nonce_string);
	apply_sha256(block_content, block_hash);
	return 0;
}

__global__ void dummy_kernel() {}

// Warm-up function
void warm_up_gpu() {
    cudaGetDeviceProperties(&deviceProp, 0);
    BYTE *dummy_data;
    cudaMalloc((void **)&dummy_data, 256);
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(dummy_data);
}
