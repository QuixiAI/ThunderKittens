// Prints writer-side vs reader-side addresses for st_int<64,64> element (0,28)
#include "kittens.cuh"
#include <cstdio>
using namespace kittens;
using st_d = st_int<64, 64>;
__global__ void probe() {
    extern __shared__ int __shm[];
    shared_allocator al(&__shm[0]);
    st_int8<64,64> (&a)[2] = al.allocate<st_int8<64,64>, 2>();
    st_int8<64,64> &b = al.allocate<st_int8<64,64>>();
    st_d (&dsm)[2] = al.allocate<st_d, 2>();
    if(threadIdx.x != 0) return;
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&dsm[0].data[0]));
    printf("dsm[0] shared base = %u (mod 1024 = %u)\n", shared_addr, shared_addr % 1024);
    printf("generic base mod 1024 = %llu\n", (unsigned long long)(uint64_t)&dsm[0].data[0] % 1024);
    // writer address for (row=0, col=20) fragment, +32B (element (0,28)):
    constexpr int subtile_cols = st_d::swizzle_bytes / sizeof(int);
    int row = 0, col = 20;
    uint32_t addr_1 = shared_addr + sizeof(int)*((col/subtile_cols)*st_d::underlying_rows*subtile_cols + row*subtile_cols + col%subtile_cols);
    int blit = 4; // lane%4==2
    int swizzle_1 = blit ^ ((addr_1 % (st_d::swizzle_bytes*8)) >> 7) << 4;
    printf("writer: value(0,29)->%u  value(0,28)->%u\n", (addr_1+32)^swizzle_1, (addr_1+36)^swizzle_1);
    // reader idx for (0,28) and (0,29):
    printf("reader: (0,28)->%u  (0,29)->%u\n", dsm[0].idx(shared_addr, {0,28}), dsm[0].idx(shared_addr, {0,29}));
    printf("swizzle_bytes=%d subtile_cols=%d\n", st_d::swizzle_bytes, subtile_cols);
}
int main() {
    cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, 64*1024);
    probe<<<1, 32, 64*1024>>>();
    cudaDeviceSynchronize();
    return 0;
}
