#include "reductions.cuh"

#ifdef TEST_GROUP_REG_TILE_REDUCTIONS

struct normalize_row {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::warp::row_sum(accum, reg_tile);
        kittens::warp::div_row(reg_tile, reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};
struct normalize_col {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::warp::col_sum(accum, reg_tile);
        kittens::warp::div_col(reg_tile, reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};
struct broadcast_row {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::warp::row_sum(accum, reg_tile);
        kittens::warp::broadcast_row(reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};
struct broadcast_col {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::warp::col_sum(accum, reg_tile);
        kittens::warp::broadcast_col(reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};
struct max_broadcast_row {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_max_broadcast_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++) {
            float row_max = i_ref[i*W*16];
            for(int j = 1; j < W*16; j++) {
                float val = i_ref[i*W*16+j];
                row_max = val > row_max ? val : row_max;
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_max;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::warp::row_max(accum, reg_tile);
        kittens::warp::broadcast_row(reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};
struct max_broadcast_col {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_max_broadcast_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*16; i++) {
            float col_max = i_ref[i];
            for(int j = 1; j < H*16; j++) {
                float val = i_ref[i+j*W*16];
                col_max = val > col_max ? val : col_max;
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_max;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::warp::col_max(accum, reg_tile);
        kittens::warp::broadcast_col(reg_tile, accum);
        kittens::warp::store(output, reg_tile, {});
    }
};

__device__ static inline bool same_float2_as_leader(float2 value, int leader) {
    float leader_x = __shfl_sync(0xffffffff, value.x, leader);
    float leader_y = __shfl_sync(0xffffffff, value.y, leader);
    return __float_as_uint(value.x) == __float_as_uint(leader_x) &&
           __float_as_uint(value.y) == __float_as_uint(leader_y);
}

template<kittens::ducks::rv::all RV>
__device__ static inline bool rv_matches_leader_bits(const RV &accum, int leader) {
    bool matches = true;
    #pragma unroll
    for(int i = 0; i < RV::outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < RV::inner_dim; j++) {
            matches &= same_float2_as_leader(accum[i][j], leader);
        }
    }
    return matches;
}

template<int H, int W, gl_t GLT>
__device__ static inline void write_bitwise_check_result(const GLT &output, bool matches) {
    // Emit a single sentinel so the usual host validator can catch any lane-level bit mismatch.
    #pragma unroll
    for(int idx = kittens::laneid(); idx < H*W*256; idx += kittens::WARP_THREADS) {
        output.raw_ptr[idx] = 0.0f;
    }
    unsigned mismatch_mask = __ballot_sync(0xffffffff, !matches);
    if(kittens::laneid() == 0) {
        output.raw_ptr[0] = mismatch_mask ? 1.0f : 0.0f;
    }
}

struct sum_bitwise_row {
    using dtype = float;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_sum_bitwise_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = 0.0f;
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::warp::row_sum(accum, reg_tile);
        int leader = std::is_same_v<L, kittens::ducks::rt_layout::row> ? (kittens::laneid() & 0x1c) : (kittens::laneid() & 0x3);
        write_bitwise_check_result<H, W>(output, rv_matches_leader_bits(accum, leader));
    }
};
struct sum_bitwise_col {
    using dtype = float;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_sum_bitwise_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = 0.0f;
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::warp::col_sum(accum, reg_tile);
        int leader = std::is_same_v<L, kittens::ducks::rt_layout::col> ? (kittens::laneid() & 0x1c) : (kittens::laneid() & 0x3);
        write_bitwise_check_result<H, W>(output, rv_matches_leader_bits(accum, leader));
    }
};
struct prod_bitwise_row {
    using dtype = float;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_prod_bitwise_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = 0.0f;
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::col_vec accum;
        kittens::warp::row_prod(accum, reg_tile);
        int leader = std::is_same_v<L, kittens::ducks::rt_layout::row> ? (kittens::laneid() & 0x1c) : (kittens::laneid() & 0x3);
        write_bitwise_check_result<H, W>(output, rv_matches_leader_bits(accum, leader));
    }
};
struct prod_bitwise_col {
    using dtype = float;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_prod_bitwise_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = 0.0f;
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::warp::load(reg_tile, input, {});
        typename kittens::rt_fl<16*H, 16*W, L>::row_vec accum;
        kittens::warp::col_prod(accum, reg_tile);
        int leader = std::is_same_v<L, kittens::ducks::rt_layout::col> ? (kittens::laneid() & 0x1c) : (kittens::laneid() & 0x3);
        write_bitwise_check_result<H, W>(output, rv_matches_leader_bits(accum, leader));
    }
};

void group::reg::tile::reductions::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/register/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<max_broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<max_broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<max_broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<max_broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<sum_bitwise_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<sum_bitwise_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<sum_bitwise_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<sum_bitwise_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<prod_bitwise_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<prod_bitwise_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<prod_bitwise_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<prod_bitwise_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    std::cout << std::endl;
}

#endif
