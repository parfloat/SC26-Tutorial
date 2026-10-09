// INSTRUMENTATION: Python entry point for the instrumented copy of PyTorch 2.6's flash-attention forward.
#include <torch/extension.h>

namespace instr_flash {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, std::optional<at::Tensor> &out_,
        std::optional<at::Tensor> &alibi_slopes_, const float p_dropout, const float softmax_scale, bool is_causal,
        int window_size_left, int window_size_right, const bool return_softmax, std::optional<at::Generator> gen_);
extern bool instr_enabled;
extern at::Tensor instr_tile, instr_split, instr_comb;
extern int instr_num_splits;
}

// q, k, v: (batch, seqlen, heads, head_dim) FP16, as aten's _flash_attention_forward passes them.
// Returns (out, softmax_lse, tile_dump, split_dump, combine_dump, num_splits).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int>
fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, bool is_causal, double scale, bool instr) {
    std::optional<at::Tensor> out, alibi;
    instr_flash::instr_enabled = instr;
    auto r = instr_flash::mha_fwd(q, k, v, out, alibi, 0.f, (float)scale, is_causal, -1, -1, false, std::nullopt);
    return {std::get<0>(r), std::get<4>(r), instr_flash::instr_tile, instr_flash::instr_split,
            instr_flash::instr_comb, instr_flash::instr_num_splits};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &fwd, "instrumented flash-attention forward"); }
