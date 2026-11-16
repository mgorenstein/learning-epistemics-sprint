// Zero-copy residual-stream activation extractor for Ollama Gemma 4 GGUF blobs via llama.cpp.
//
// Ollama bundles 356 multimodal vision tensors ("v.*" and "mm.*") alongside the 658 Gemma 4
// text tensors inside a single GGUF file. This tool wraps the mmap'd Ollama blob in a virtual
// FILE* (funopen) that strips the vision metadata/tensors in memory — requiring zero extra disk
// space — and extracts all per-layer residual-stream states at the final prompt token, reusing
// the KV cache across trials within each block.
//
// Build (macOS with Homebrew llama.cpp):
//   clang++ -O3 -std=c++17 tools/extract_gemma4_acts.cpp \
//     -I/opt/homebrew/opt/llama.cpp/include -I/opt/homebrew/opt/ggml/include \
//     -L/opt/homebrew/opt/llama.cpp/lib -L/opt/homebrew/opt/ggml/lib \
//     -Wl,-rpath,/opt/homebrew/opt/llama.cpp/lib -Wl,-rpath,/opt/homebrew/opt/ggml/lib \
//     -lllama -lggml -lggml-base -o tools/extract_gemma4_acts
//
// Usage:
//   ./tools/extract_gemma4_acts <ollama_gguf_blob> <prompts.bin> <acts_out.f32.bin> <logits_out.csv>

#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fprintf(stderr, "%s", text);
}

struct TensorSpan { uint64_t v_begin, v_end; const uint8_t * src_ptr; };
struct VirtualGGUF { std::vector<uint8_t> meta_buf; std::vector<TensorSpan> spans; uint64_t total_size = 0, pos = 0; };

static int vgguf_read(void * cookie, char * buf, int nbytes) {
    auto * vg = static_cast<VirtualGGUF *>(cookie);
    if (nbytes <= 0 || vg->pos >= vg->total_size) return 0;
    uint64_t rem = std::min<uint64_t>((uint64_t)nbytes, vg->total_size - vg->pos);
    uint64_t copied = 0;
    while (copied < rem) {
        uint64_t cur = vg->pos + copied;
        uint64_t meta_sz = vg->meta_buf.size();
        if (cur < meta_sz) {
            uint64_t take = std::min<uint64_t>(rem - copied, meta_sz - cur);
            std::memcpy(buf + copied, vg->meta_buf.data() + cur, take);
            copied += take;
            continue;
        }
        auto it = std::upper_bound(vg->spans.begin(), vg->spans.end(), cur,
            [](uint64_t val, const TensorSpan & s) { return val < s.v_end; });
        if (it == vg->spans.end()) {
            uint64_t take = rem - copied;
            std::memset(buf + copied, 0, take);
            copied += take;
        } else if (cur < it->v_begin) {
            uint64_t take = std::min<uint64_t>(rem - copied, it->v_begin - cur);
            std::memset(buf + copied, 0, take);
            copied += take;
        } else {
            uint64_t take = std::min<uint64_t>(rem - copied, it->v_end - cur);
            std::memcpy(buf + copied, it->src_ptr + (cur - it->v_begin), take);
            copied += take;
        }
    }
    vg->pos += copied;
    return (int)copied;
}

static fpos_t vgguf_seek(void * cookie, fpos_t offset, int whence) {
    auto * vg = static_cast<VirtualGGUF *>(cookie);
    int64_t target = (whence == SEEK_SET)
        ? offset
        : (whence == SEEK_CUR ? (int64_t)vg->pos + offset : (int64_t)vg->total_size + offset);
    if (target < 0 || (uint64_t)target > vg->total_size) return -1;
    vg->pos = (uint64_t)target;
    return (fpos_t)vg->pos;
}

static llama_model * load_ollama_gemma4_text(
    const char * blob_path, void ** out_mmap, size_t * out_mmap_size, int * out_fd
) {
    int fd = ::open(blob_path, O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    if (::fstat(fd, &st) != 0) { ::close(fd); return nullptr; }
    void * mapped = ::mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) { ::close(fd); return nullptr; }
    *out_mmap = mapped;
    *out_mmap_size = (size_t)st.st_size;
    *out_fd = fd;

    ggml_context * src_tensors = nullptr;
    gguf_init_params gp = { true, &src_tensors };
    gguf_context * src_gguf = gguf_init_from_file(blob_path, gp);
    if (!src_gguf) return nullptr;

    gguf_context * clean_gguf = gguf_init_empty();
    gguf_set_kv(clean_gguf, src_gguf);
    std::vector<std::string> rm_keys;
    for (int64_t i = 0; i < gguf_get_n_kv(clean_gguf); ++i) {
        const char * k = gguf_get_key(clean_gguf, i);
        if (std::strncmp(k, "gemma4.vision.", 14) == 0) rm_keys.emplace_back(k);
    }
    for (const auto & k : rm_keys) gguf_remove_key(clean_gguf, k.c_str());

    std::vector<std::string> kept_names;
    for (ggml_tensor * t = ggml_get_first_tensor(src_tensors); t != nullptr; t = ggml_get_next_tensor(src_tensors, t)) {
        const char * name = ggml_get_name(t);
        if (std::strncmp(name, "v.", 2) == 0 || std::strncmp(name, "mm.", 3) == 0) continue;
        gguf_add_tensor(clean_gguf, t);
        kept_names.emplace_back(name);
    }

    VirtualGGUF vg;
    size_t meta_sz = gguf_get_meta_size(clean_gguf);
    vg.meta_buf.resize(meta_sz);
    gguf_get_meta_data(clean_gguf, vg.meta_buf.data());
    const uint8_t * src_base = static_cast<const uint8_t *>(mapped);
    size_t src_data_off = gguf_get_data_offset(src_gguf);
    uint64_t max_end = meta_sz;
    for (size_t i = 0; i < kept_names.size(); ++i) {
        int64_t clean_id = (int64_t)i;
        int64_t src_id = gguf_find_tensor(src_gguf, kept_names[i].c_str());
        uint64_t v_beg = meta_sz + gguf_get_tensor_offset(clean_gguf, clean_id);
        uint64_t sz = gguf_get_tensor_size(clean_gguf, clean_id);
        const uint8_t * s_ptr = src_base + src_data_off + gguf_get_tensor_offset(src_gguf, src_id);
        vg.spans.push_back({v_beg, v_beg + sz, s_ptr});
        if (v_beg + sz > max_end) max_end = v_beg + sz;
    }
    vg.total_size = max_end;

    FILE * vf = ::funopen(&vg, vgguf_read, nullptr, vgguf_seek, nullptr);
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = -1;
    mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file_ptr(vf, mparams);
    std::fclose(vf);
    gguf_free(clean_gguf);
    gguf_free(src_gguf);
    ggml_free(src_tensors);
    return model;
}

struct FinalNormCapture { std::vector<float> result_norm; int n_embd = 0; };

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (std::strcmp(ggml_get_name(t), "result_norm") != 0) return false;
    if (ask) return true;
    auto * cap = static_cast<FinalNormCapture *>(user_data);
    int64_t n_rows = t->ne[1];
    cap->result_norm.resize(cap->n_embd);
    ggml_backend_tensor_get(t, cap->result_norm.data(),
        (size_t)(n_rows - 1) * cap->n_embd * sizeof(float),
        cap->n_embd * sizeof(float));
    return true;
}

static std::string clean_word_piece(const char * s, int len) {
    int a = 0, b = len;
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n')) a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n')) b--;
    std::string out;
    out.reserve(b - a);
    for (int i = a; i < b; ++i) out.push_back((char)std::tolower((unsigned char)s[i]));
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "Usage: %s <model_blob> <prompts.bin> <acts_out.f32.bin> <logits_out.csv>\n", argv[0]);
        return 1;
    }
    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    auto t_start = std::chrono::steady_clock::now();
    void * mapped = nullptr;
    size_t mapped_sz = 0;
    int fd = -1;
    llama_model * model = load_ollama_gemma4_text(argv[1], &mapped, &mapped_sz, &fd);
    if (!model) {
        std::fprintf(stderr, "Failed to load model from %s\n", argv[1]);
        return 1;
    }

    int n_layer = llama_model_n_layer(model);
    int n_embd = llama_model_n_embd(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    std::printf("Model loaded: n_layer=%d (+1 final norm = %d), n_embd=%d, n_vocab=%d\n",
                n_layer, n_layer + 1, n_embd, n_vocab);
    std::fflush(stdout);

    const char * target_words[12] = {
        "red", "blue", "green", "yellow",
        "square", "circle", "triangle", "diamond",
        "color", "shape", "not", "guessing"
    };
    std::vector<int> word_tok_ids[12];
    char pbuf[128];
    for (int id = 0; id < n_vocab; ++id) {
        if (llama_vocab_is_control(vocab, id)) continue;
        int plen = llama_token_to_piece(vocab, id, pbuf, sizeof(pbuf) - 1, 0, false);
        if (plen <= 0 || plen > 24) continue;
        std::string cw = clean_word_piece(pbuf, plen);
        for (int w = 0; w < 12; ++w) {
            if (cw == target_words[w]) word_tok_ids[w].push_back(id);
        }
    }

    FinalNormCapture cap{ {}, n_embd };
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    cparams.embeddings = false;
    cparams.cb_eval = eval_cb;
    cparams.cb_eval_user_data = &cap;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;
    for (int lid = 0; lid < n_layer; ++lid) {
        llama_set_embeddings_layer_inp(ctx, lid, true);
    }

    FILE * fin = std::fopen(argv[2], "rb");
    FILE * fout_acts = std::fopen(argv[3], "wb");
    FILE * fout_csv = std::fopen(argv[4], "w");
    if (!fin || !fout_acts || !fout_csv) {
        std::fprintf(stderr, "Failed to open I/O files\n");
        return 1;
    }

    uint32_t n_records = 0;
    if (std::fread(&n_records, sizeof(uint32_t), 1, fin) != 1) return 1;
    std::printf("Extracting residual-stream activations for %u trials...\n", n_records);
    std::fflush(stdout);

    std::fprintf(fout_csv, "row_idx,n_tok,n_new,top1_id,top1_prob,p_red,p_blue,p_green,p_yellow,p_square,p_circle,p_triangle,p_diamond,p_color,p_shape,p_not,p_guessing\n");

    std::vector<llama_token> prev_tokens;
    std::vector<llama_token> cur_tokens(2048);
    std::string prompt_buf;
    auto t_loop0 = std::chrono::steady_clock::now();

    for (uint32_t rec = 0; rec < n_records; ++rec) {
        uint32_t hdr[2] = {0, 0};
        if (std::fread(hdr, sizeof(uint32_t), 2, fin) != 2) return 1;
        uint32_t is_new_block = hdr[0];
        uint32_t byte_len = hdr[1];
        prompt_buf.resize(byte_len);
        if (std::fread(prompt_buf.data(), 1, byte_len, fin) != byte_len) return 1;

        cur_tokens.resize(2048);
        int n_tok = llama_tokenize(vocab, prompt_buf.c_str(), (int)prompt_buf.size(),
                                   cur_tokens.data(), (int)cur_tokens.size(), false, true);
        if (n_tok <= 0 || n_tok >= 2048) {
            std::fprintf(stderr, "Tokenize overflow at rec %u: n_tok=%d\n", rec, n_tok);
            return 1;
        }
        cur_tokens.resize(n_tok);

        int common = 0;
        if (is_new_block) {
            llama_memory_clear(llama_get_memory(ctx), false);
            prev_tokens.clear();
        } else {
            while (common < (int)prev_tokens.size() && common < n_tok && prev_tokens[common] == cur_tokens[common]) {
                common++;
            }
            if (common >= n_tok) common = n_tok - 1;
            llama_memory_seq_rm(llama_get_memory(ctx), 0, common, -1);
        }

        int n_new = n_tok - common;
        llama_batch batch = llama_batch_init(n_new, 0, 1);
        batch.n_tokens = n_new;
        for (int i = 0; i < n_new; ++i) {
            batch.token[i] = cur_tokens[common + i];
            batch.pos[i] = common + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_new - 1) ? 1 : 0;
        }

        int rc = llama_decode(ctx, batch);
        llama_synchronize(ctx);
        if (rc != 0) {
            std::fprintf(stderr, "llama_decode failed at rec %u with rc=%d\n", rec, rc);
            return 1;
        }

        // L00..L29 residual stream entering blk.lid + L30 final RMSNorm output
        for (int lid = 0; lid < n_layer; ++lid) {
            const float * row_ptr = llama_get_embeddings_layer_inp(ctx, lid) + (size_t)(n_new - 1) * n_embd;
            std::fwrite(row_ptr, sizeof(float), n_embd, fout_acts);
        }
        std::fwrite(cap.result_norm.data(), sizeof(float), n_embd, fout_acts);

        const float * logits = llama_get_logits_ith(ctx, -1);
        int top1_id = 0;
        float max_l = logits[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                top1_id = i;
            }
        }
        double sum_exp = 0.0;
        for (int i = 0; i < n_vocab; ++i) sum_exp += std::exp((double)logits[i] - (double)max_l);
        double top1_prob = std::exp((double)logits[top1_id] - (double)max_l) / sum_exp;
        double w_probs[12] = {0};
        for (int w = 0; w < 12; ++w) {
            double s = 0.0;
            for (int id : word_tok_ids[w]) s += std::exp((double)logits[id] - (double)max_l);
            w_probs[w] = s / sum_exp;
        }

        std::fprintf(fout_csv,
            "%u,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            rec, n_tok, n_new, top1_id, top1_prob,
            w_probs[0], w_probs[1], w_probs[2], w_probs[3],
            w_probs[4], w_probs[5], w_probs[6], w_probs[7],
            w_probs[8], w_probs[9], w_probs[10], w_probs[11]);

        llama_batch_free(batch);
        prev_tokens = cur_tokens;

        if ((rec + 1) % 160 == 0 || rec + 1 == n_records) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_loop0).count();
            double rate = (rec + 1) / elapsed;
            double eta = (n_records - (rec + 1)) / rate;
            std::printf("  [%4u / %4u] %.1fs elapsed (%.2f trials/s, ETA %.1fs)\n",
                        rec + 1, n_records, elapsed, rate, eta);
            std::fflush(stdout);
            std::fflush(fout_csv);
        }
    }

    std::fclose(fin);
    std::fclose(fout_acts);
    std::fclose(fout_csv);

    auto t_end = std::chrono::steady_clock::now();
    std::printf("Done! Total wall time: %.1fs\n",
                std::chrono::duration<double>(t_end - t_start).count());

    llama_free(ctx);
    llama_model_free(model);
    ::munmap(mapped, mapped_sz);
    ::close(fd);
    llama_backend_free();
    return 0;
}
