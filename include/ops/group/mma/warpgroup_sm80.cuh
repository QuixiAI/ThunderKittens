/**
 * @file
 * @brief Ampere (SM80/SM86) emulation of the Hopper warpgroup (WGMMA) MMA operations.
 *
 * Same call signatures as warpgroup.cuh, so consumer kernels compile unchanged.
 * WGMMA's warpgroup-wide m64 instruction distributes 64 output rows as 16 rows
 * per warp; this emulation has each warp compute its 16-row slice directly with
 * warp-scope mma.sync (m16n8k16 / m16n8k32) base ops, loading shared-memory
 * operand fragments via ldmatrix. mma.sync is synchronous, so the WGMMA
 * fence/commit/wait functions become no-ops with identical call-site semantics.
 *
 * Note: fp8 operands are not supported here (Ampere has no fp8 MMA); fp8 on
 * Ampere is a storage format — dequantize to bf16/half in registers first.
 */

//  ------------------------------------------------------ FENCES ------------------------------------------------------
// mma.sync completes synchronously; all of these are no-ops on Ampere.

template<ducks::rt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP
}
template<ducks::crt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP
}
template<typename T=kittens::ducks::default_type> // prevents static assert being instantiated unless called.
__device__ static inline void mma_fence() {
    KITTENS_CHECK_WARPGROUP
}
template<typename T=kittens::ducks::default_type>
__device__ static inline void mma_commit_group() {
    KITTENS_CHECK_WARPGROUP
}
template<int N=0>
__device__ static inline void mma_async_wait() {
    KITTENS_CHECK_WARPGROUP
}

//  --------------------------------------------------- PLUMBING -------------------------------------------------------

/**
 * @brief A minimal shared-tile chunk view for the WGMMA emulation.
 *
 * Quacks like an st (same idx math as st_subtile) but is constructible from a
 * raw data pointer plus element offsets, so chunks can be taken of both full
 * shared tiles and st_subtile views.
 */
template<typename _ST, int _rows, int _cols>
struct sm80_st_chunk {
    using identifier = ducks::st::identifier;
    using ST = _ST;
    using T = typename ST::T;
    using T2 = typename ST::T2;
    using dtype = T;

    static constexpr bool swizzle = ST::swizzle;
    static constexpr int underlying_rows         = ST::underlying_rows;
    static constexpr int underlying_cols         = ST::underlying_cols;
    static constexpr int underlying_num_elements = ST::underlying_num_elements;
    static constexpr int rows         = _rows;
    static constexpr int cols         = _cols;
    static constexpr int num_elements = rows * cols;
    static constexpr int swizzle_bytes = ST::swizzle_bytes;

    dtype *data;
    int row_offset, col_offset;

    __device__ sm80_st_chunk(dtype *_data, int _row_offset, int _col_offset)
        : data(_data), row_offset(_row_offset), col_offset(_col_offset) {}

    __device__ inline uint32_t idx(uint32_t ptr, const int2 coord) const {
        int r = coord.x+row_offset, c = coord.y+col_offset;
        if constexpr(swizzle) {
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            const uint32_t addr = ptr + sizeof(T)*(outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols);
            const int swz = ((addr % swizzle_repeat) >> 7) << 4;
            return (addr ^ swz);
        } else {
            return ptr + sizeof(T)*(r*cols + c);
        }
    }
    __device__ inline T* idx(T *ptr, const int2 coord) const {
        int r = coord.x+row_offset, c = coord.y+col_offset;
        if constexpr(swizzle) {
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swz = ((addr % swizzle_repeat) >> 7) << 4;
            return (T*)(addr ^ swz);
        } else {
            return &ptr[r*cols + c];
        }
    }
    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol);
    }
};

template<typename S> static constexpr bool sm80_is_st_subtile = requires { typename S::ST; };

template<int RD, int CD, ducks::st::all S>
__device__ static inline auto sm80_chunk(const S &s, int r, int c) {
    if constexpr(sm80_is_st_subtile<S>) {
        return sm80_st_chunk<typename S::ST, RD, CD>(
            const_cast<typename S::dtype*>(&s.data[0]), s.row_offset + r*RD, s.col_offset + c*CD);
    } else {
        return sm80_st_chunk<S, RD, CD>(
            const_cast<typename S::dtype*>(&s.data[0]), r*RD, c*CD);
    }
}

/// In-place negation of one register base tile (for the complex-mma sign flip).
template<typename RTB>
__device__ static inline void sm80_neg(RTB &frag) {
    #pragma unroll
    for(int p = 0; p < RTB::packed_per_tile; p++) {
        frag.data[p] = base_ops::sub::template op<typename RTB::dtype>(
            base_types::constants<typename RTB::dtype>::zero(), frag.data[p]);
    }
}

#define KITTENS_SM80_WGMMA_AB_TYPES \
    static_assert( \
        (std::is_same_v<typename D::T, float> && std::is_same_v<T_AB, bf16>) || \
        (std::is_same_v<typename D::T, float> && std::is_same_v<T_AB, half>) || \
        (std::is_same_v<typename D::T, half>  && std::is_same_v<T_AB, half>), \
        "SM80 warpgroup mma: unsupported type combination (fp8 needs dequant to bf16/half; int8 only via mma_ABt).");
#define KITTENS_SM80_WGMMA_ABT_TYPES \
    static_assert( \
        (std::is_same_v<typename D::T, float> && std::is_same_v<T_AB, bf16>) || \
        (std::is_same_v<typename D::T, float> && std::is_same_v<T_AB, half>) || \
        (std::is_same_v<typename D::T, half>  && std::is_same_v<T_AB, half>) || \
        (std::is_same_v<typename D::T, int>   && (std::is_same_v<T_AB, int8> || std::is_same_v<T_AB, uint8>)), \
        "SM80 warpgroup mma_ABt: unsupported type combination (fp8 needs dequant to bf16/half).");

//  ---------------------------------------------------- WORKERS -------------------------------------------------------
// Each worker computes this warp's 16-row slice of the warpgroup-wide result.
// neg_a flips the sign of the A operand (used by the complex composition).

// [(register, shared) -> register]
template<int neg_a, int accumulate, ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B>
__device__ static inline void sm80_mma_AB_rt(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM80_WGMMA_AB_TYPES
    static_assert(std::is_same_v<typename B::T, T_AB>);
    constexpr int RD = kittens::TILE_ROW_DIM<T_AB>, CD = kittens::TILE_COL_DIM<T_AB>;
    constexpr int M = A::height;
    static_assert(D::height == M);
    constexpr int K = A::width;
    constexpr int N = D::width;
    static_assert(B::rows == K*RD && B::cols == N*CD);
    if constexpr(!accumulate) ::kittens::group<1>::zero(d);
    #pragma unroll
    for(int k = 0; k < K; k++) {
        #pragma unroll
        for(int j = 0; j < N; j++) {
            rt<T_AB, RD, CD, ducks::rt_layout::col> b_frag;
            ::kittens::group<1>::load(b_frag, sm80_chunk<RD, CD>(b, k, j));
            #pragma unroll
            for(int m = 0; m < M; m++) {
                if constexpr(neg_a) {
                    auto a_neg = a.tiles[m][k];
                    sm80_neg(a_neg);
                    mma_AB_base(d.tiles[m][j], a_neg, b_frag.tiles[0][0], d.tiles[m][j]);
                } else {
                    mma_AB_base(d.tiles[m][j], a.tiles[m][k], b_frag.tiles[0][0], d.tiles[m][j]);
                }
            }
        }
    }
}

// [(register, shared) -> register], B transposed (B is N x K)
template<int neg_a, int accumulate, ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B>
__device__ static inline void sm80_mma_ABt_rt(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM80_WGMMA_ABT_TYPES
    static_assert(std::is_same_v<typename B::T, T_AB>);
    constexpr int RD = kittens::TILE_ROW_DIM<T_AB>, CD = kittens::TILE_COL_DIM<T_AB>;
    constexpr int M = A::height;
    static_assert(D::height == M);
    constexpr int K = A::width;
    constexpr int N = D::width;
    static_assert(B::rows == N*RD && B::cols == K*CD);
    if constexpr(!accumulate) ::kittens::group<1>::zero(d);
    #pragma unroll
    for(int k = 0; k < K; k++) {
        #pragma unroll
        for(int j = 0; j < N; j++) {
            rt<T_AB, RD, CD, ducks::rt_layout::row> b_frag;
            ::kittens::group<1>::load(b_frag, sm80_chunk<RD, CD>(b, j, k));
            #pragma unroll
            for(int m = 0; m < M; m++) {
                if constexpr(neg_a) {
                    auto a_neg = a.tiles[m][k];
                    sm80_neg(a_neg);
                    mma_ABt_base(d.tiles[m][j], a_neg, b_frag.tiles[0][0], d.tiles[m][j]);
                } else {
                    mma_ABt_base(d.tiles[m][j], a.tiles[m][k], b_frag.tiles[0][0], d.tiles[m][j]);
                }
            }
        }
    }
}

// [(shared, shared) -> register]
template<int neg_a, int accumulate, ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void sm80_mma_AB_st(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM80_WGMMA_AB_TYPES
    static_assert(std::is_same_v<typename B::T, T_AB>);
    constexpr int RD = kittens::TILE_ROW_DIM<T_AB>, CD = kittens::TILE_COL_DIM<T_AB>;
    static_assert(A::rows / RD == GROUP_WARPS && A::rows % RD == 0); // logical M == 64
    static_assert(D::height == 1);
    constexpr int K = A::cols / CD;
    constexpr int N = D::width;
    static_assert(B::rows == K*RD && B::cols == N*CD);
    const int w = warpid();
    if constexpr(!accumulate) ::kittens::group<1>::zero(d);
    #pragma unroll
    for(int k = 0; k < K; k++) {
        rt<T_AB, RD, CD, ducks::rt_layout::row> a_frag;
        ::kittens::group<1>::load(a_frag, sm80_chunk<RD, CD>(a, w, k));
        if constexpr(neg_a) sm80_neg(a_frag.tiles[0][0]);
        #pragma unroll
        for(int j = 0; j < N; j++) {
            rt<T_AB, RD, CD, ducks::rt_layout::col> b_frag;
            ::kittens::group<1>::load(b_frag, sm80_chunk<RD, CD>(b, k, j));
            mma_AB_base(d.tiles[0][j], a_frag.tiles[0][0], b_frag.tiles[0][0], d.tiles[0][j]);
        }
    }
}

// [(shared, shared) -> register], B transposed (B is N x K)
template<int neg_a, int accumulate, ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void sm80_mma_ABt_st(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM80_WGMMA_ABT_TYPES
    static_assert(std::is_same_v<typename B::T, T_AB>);
    constexpr int RD = kittens::TILE_ROW_DIM<T_AB>, CD = kittens::TILE_COL_DIM<T_AB>;
    static_assert(A::rows / RD == GROUP_WARPS && A::rows % RD == 0);
    static_assert(D::height == 1);
    constexpr int K = A::cols / CD;
    constexpr int N = D::width;
    static_assert(B::rows == N*RD && B::cols == K*CD);
    const int w = warpid();
    if constexpr(!accumulate) ::kittens::group<1>::zero(d);
    #pragma unroll
    for(int k = 0; k < K; k++) {
        rt<T_AB, RD, CD, ducks::rt_layout::row> a_frag;
        ::kittens::group<1>::load(a_frag, sm80_chunk<RD, CD>(a, w, k));
        if constexpr(neg_a) sm80_neg(a_frag.tiles[0][0]);
        #pragma unroll
        for(int j = 0; j < N; j++) {
            rt<T_AB, RD, CD, ducks::rt_layout::row> b_frag;
            ::kittens::group<1>::load(b_frag, sm80_chunk<RD, CD>(b, j, k));
            mma_ABt_base(d.tiles[0][j], a_frag.tiles[0][0], b_frag.tiles[0][0], d.tiles[0][j]);
        }
    }
}

// [(shared, shared) -> register], A transposed (A is K x 64)
template<int neg_a, int accumulate, ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void sm80_mma_AtB_st(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM80_WGMMA_AB_TYPES
    static_assert(std::is_same_v<typename B::T, T_AB>);
    constexpr int RD = kittens::TILE_ROW_DIM<T_AB>, CD = kittens::TILE_COL_DIM<T_AB>;
    static_assert(A::cols / CD == GROUP_WARPS && A::cols % CD == 0); // logical M == 64
    static_assert(D::height == 1);
    constexpr int K = A::rows / RD;
    constexpr int N = D::width;
    static_assert(B::rows == K*RD && B::cols == N*CD);
    const int w = warpid();
    if constexpr(!accumulate) ::kittens::group<1>::zero(d);
    #pragma unroll
    for(int k = 0; k < K; k++) {
        rt<T_AB, RD, CD, ducks::rt_layout::col> a_frag;
        ::kittens::group<1>::load(a_frag, sm80_chunk<RD, CD>(a, k, w));
        if constexpr(neg_a) sm80_neg(a_frag.tiles[0][0]);
        #pragma unroll
        for(int j = 0; j < N; j++) {
            rt<T_AB, RD, CD, ducks::rt_layout::col> b_frag;
            ::kittens::group<1>::load(b_frag, sm80_chunk<RD, CD>(b, k, j));
            mma_AtB_base(d.tiles[0][j], a_frag.tiles[0][0], b_frag.tiles[0][0], d.tiles[0][j]);
        }
    }
}

// [(shared, shared) -> register], A and B transposed
template<int neg_a, int accumulate, ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void sm80_mma_AtBt_st(D &d, const A &a, const B &b) {
    KITTENS_CHECK_WARPGROUP
    using T_AB = typename A::T;
    KITTENS_SM80_WGMMA_AB_TYPES
    static_assert(std::is_same_v<typename B::T, T_AB>);
    constexpr int RD = kittens::TILE_ROW_DIM<T_AB>, CD = kittens::TILE_COL_DIM<T_AB>;
    static_assert(A::cols / CD == GROUP_WARPS && A::cols % CD == 0);
    static_assert(D::height == 1);
    constexpr int K = A::rows / RD;
    constexpr int N = D::width;
    static_assert(B::rows == N*RD && B::cols == K*CD);
    const int w = warpid();
    if constexpr(!accumulate) ::kittens::group<1>::zero(d);
    #pragma unroll
    for(int k = 0; k < K; k++) {
        rt<T_AB, RD, CD, ducks::rt_layout::col> a_frag;
        ::kittens::group<1>::load(a_frag, sm80_chunk<RD, CD>(a, k, w));
        if constexpr(neg_a) sm80_neg(a_frag.tiles[0][0]);
        #pragma unroll
        for(int j = 0; j < N; j++) {
            rt<T_AB, RD, CD, ducks::rt_layout::row> b_frag;
            ::kittens::group<1>::load(b_frag, sm80_chunk<RD, CD>(b, j, k));
            mma_AtBt_base(d.tiles[0][j], a_frag.tiles[0][0], b_frag.tiles[0][0], d.tiles[0][j]);
        }
    }
}

//  ------------------------------------------------- PUBLIC API (REAL) ------------------------------------------------
// Signatures match warpgroup.cuh; the fence template parameter is accepted and ignored.

template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    sm80_mma_AB_rt<0, accumulate>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    sm80_mma_AB_rt<0, 0>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    sm80_mma_AB_st<0, accumulate>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    sm80_mma_AB_st<0, 0>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    sm80_mma_ABt_rt<0, accumulate>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st::all B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    sm80_mma_ABt_rt<0, 0>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    sm80_mma_ABt_st<0, accumulate>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    sm80_mma_ABt_st<0, 0>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
    sm80_mma_AtB_st<0, accumulate>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    sm80_mma_AtB_st<0, 0>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    sm80_mma_AtBt_st<0, accumulate>(d, a, b);
}
template<ducks::rt::row_layout D, ducks::st::all A, ducks::st::all B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    sm80_mma_AtBt_st<0, 0>(d, a, b);
}

//  ----------------------------------------------- PUBLIC API (COMPLEX) -----------------------------------------------
// Composed from the real workers: (ar + i*ai)(br + i*bi) = (ar*br - ai*bi) + i*(ar*bi + ai*br)

template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::cst::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    sm80_mma_AB_rt<0, accumulate>(d.real, a.real, b.real);
    sm80_mma_AB_rt<1, 1>(d.real, a.imag, b.imag);
    sm80_mma_AB_rt<0, accumulate>(d.imag, a.real, b.imag);
    sm80_mma_AB_rt<0, 1>(d.imag, a.imag, b.real);
}
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::cst::all B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}
template<ducks::crt::row_layout D, ducks::cst::all A, ducks::cst::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    sm80_mma_AB_st<0, accumulate>(d.real, a.real, b.real);
    sm80_mma_AB_st<1, 1>(d.real, a.imag, b.imag);
    sm80_mma_AB_st<0, accumulate>(d.imag, a.real, b.imag);
    sm80_mma_AB_st<0, 1>(d.imag, a.imag, b.real);
}
template<ducks::crt::row_layout D, ducks::cst::all A, ducks::cst::all B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::cst::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    sm80_mma_ABt_rt<0, accumulate>(d.real, a.real, b.real);
    sm80_mma_ABt_rt<1, 1>(d.real, a.imag, b.imag);
    sm80_mma_ABt_rt<0, accumulate>(d.imag, a.real, b.imag);
    sm80_mma_ABt_rt<0, 1>(d.imag, a.imag, b.real);
}
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::cst::all B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}
template<ducks::crt::row_layout D, ducks::cst::all A, ducks::cst::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    sm80_mma_ABt_st<0, accumulate>(d.real, a.real, b.real);
    sm80_mma_ABt_st<1, 1>(d.real, a.imag, b.imag);
    sm80_mma_ABt_st<0, accumulate>(d.imag, a.real, b.imag);
    sm80_mma_ABt_st<0, 1>(d.imag, a.imag, b.real);
}
template<ducks::crt::row_layout D, ducks::cst::all A, ducks::cst::all B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}
template<ducks::crt::row_layout D, ducks::cst::all A, ducks::cst::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
    sm80_mma_AtB_st<0, accumulate>(d.real, a.real, b.real);
    sm80_mma_AtB_st<1, 1>(d.real, a.imag, b.imag);
    sm80_mma_AtB_st<0, accumulate>(d.imag, a.real, b.imag);
    sm80_mma_AtB_st<0, 1>(d.imag, a.imag, b.real);
}
template<ducks::crt::row_layout D, ducks::cst::all A, ducks::cst::all B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    mma_AtB<D, A, B, 1, 0>(d, a, b);
}
template<ducks::crt::row_layout D, ducks::cst::all A, ducks::cst::all B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    sm80_mma_AtBt_st<0, accumulate>(d.real, a.real, b.real);
    sm80_mma_AtBt_st<1, 1>(d.real, a.imag, b.imag);
    sm80_mma_AtBt_st<0, accumulate>(d.imag, a.real, b.imag);
    sm80_mma_AtBt_st<0, 1>(d.imag, a.imag, b.real);
}
template<ducks::crt::row_layout D, ducks::cst::all A, ducks::cst::all B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    mma_AtBt<D, A, B, 1, 0>(d, a, b);
}

//  -------------------------------------------------- PRETTY WRAPPERS -------------------------------------------------

template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mma(D &d,
                                  const A &a,
                                  const B &b) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b);
        } else {
            mma_AtB(d, a, b);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b);
        } else {
            mma_AB(d, a, b);
        }
    }
}
template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mm(D &d,
                                  const A &a,
                                  const B &b) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mm_AtBt(d, a, b);
        } else {
            mm_AtB(d, a, b);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mm_ABt(d, a, b);
        } else {
            mm_AB(d, a, b);
        }
    }
}

#undef KITTENS_SM80_WGMMA_AB_TYPES
#undef KITTENS_SM80_WGMMA_ABT_TYPES
