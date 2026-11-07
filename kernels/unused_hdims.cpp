#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include "flash.h"
#include "namespace_config.h"

namespace FLASH_NAMESPACE {

#define FA_STUB(T, HD, CAUSAL) \
    template <> void run_mha_fwd_<T, HD, CAUSAL>(Flash_fwd_params&, cudaStream_t) {} \
    template <> void run_mha_fwd_splitkv_dispatch<T, HD, CAUSAL>(Flash_fwd_params&, cudaStream_t) {}

#define FA_STUB_BOTH(T, HD) FA_STUB(T, HD, false) FA_STUB(T, HD, true)
#define FA_STUB_DTYPES(HD)  FA_STUB_BOTH(cutlass::half_t, HD) FA_STUB_BOTH(cutlass::bfloat16_t, HD)

FA_STUB_DTYPES(32)
FA_STUB_DTYPES(64)
FA_STUB_DTYPES(96)
FA_STUB_DTYPES(160)
FA_STUB_DTYPES(192)
FA_STUB_DTYPES(256)
FA_STUB_BOTH(cutlass::half_t, 128) 

}
