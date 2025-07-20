/* Gemmini-accelerated Inference for Llama-2 Transformer model */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include "gemmini.h"
#include "gemmini_nn.h"

// ----------------------------------------------------------------------------
// Transformer model structures (same as original)

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    elem_t* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    elem_t* rms_att_weight; // (layer, dim) rmsnorm weights
    elem_t* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    elem_t* wq; // (layer, dim, n_heads * head_size)
    elem_t* wk; // (layer, dim, n_kv_heads * head_size)
    elem_t* wv; // (layer, dim, n_kv_heads * head_size)
    elem_t* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    elem_t* w1; // (layer, hidden_dim, dim)
    elem_t* w2; // (layer, dim, hidden_dim)
    elem_t* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    elem_t* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    elem_t* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    elem_t *x; // activation at current time stamp (dim,)
    elem_t *xb; // same, but inside a residual branch (dim,)
    elem_t *xb2; // an additional buffer just for convenience (dim,)
    elem_t *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    elem_t *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    elem_t *q; // query (dim,)
    elem_t *k; // key (dim,)
    elem_t *v; // value (dim,)
    elem_t *att; // buffer for scores/attention values (n_heads, seq_len)
    elem_t *logits; // output logits
    // kv cache
    elem_t* key_cache;   // (layer, seq_len, dim)
    elem_t* value_cache; // (layer, seq_len, dim)
    // Gemmini-specific buffers
    acc_t *x_acc; // accumulator buffer
    acc_t *xb_acc; // accumulator buffer
    acc_t *xb2_acc; // accumulator buffer
    acc_t *hb_acc; // accumulator buffer
    acc_t *logits_acc; // accumulator buffer
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

// ----------------------------------------------------------------------------
// Utility functions for float to elem_t conversion

void float_to_elem(const float* src, elem_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        // Assuming elem_t can handle the range, scale if needed
        dst[i] = (elem_t)(src[i] * ELEM_SCALE);
    }
}

void elem_to_float(const elem_t* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (float)src[i] / ELEM_SCALE;
    }
}

void acc_to_float(const acc_t* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (float)src[i] / (ELEM_SCALE * ELEM_SCALE);
    }
}

// ----------------------------------------------------------------------------
// Memory allocation for Gemmini

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(elem_t));
    s->xb = calloc(p->dim, sizeof(elem_t));
    s->xb2 = calloc(p->dim, sizeof(elem_t));
    s->hb = calloc(p->hidden_dim, sizeof(elem_t));
    s->hb2 = calloc(p->hidden_dim, sizeof(elem_t));
    s->q = calloc(p->dim, sizeof(elem_t));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(elem_t));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(elem_t));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(elem_t));
    s->logits = calloc(p->vocab_size, sizeof(elem_t));
    
    // Accumulator buffers
    s->x_acc = calloc(p->dim, sizeof(acc_t));
    s->xb_acc = calloc(p->dim, sizeof(acc_t));
    s->xb2_acc = calloc(p->dim, sizeof(acc_t));
    s->hb_acc = calloc(p->hidden_dim, sizeof(acc_t));
    s->logits_acc = calloc(p->vocab_size, sizeof(acc_t));
    
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits
     || !s->x_acc || !s->xb_acc || !s->xb2_acc || !s->hb_acc || !s->logits_acc) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    free(s->x_acc);
    free(s->xb_acc);
    free(s->xb2_acc);
    free(s->hb_acc);
    free(s->logits_acc);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    
    // Allocate elem_t versions of weights
    unsigned long long total_weights = 0;
    total_weights += p->vocab_size * p->dim; // token_embedding_table
    total_weights += n_layers * p->dim; // rms_att_weight
    total_weights += n_layers * p->dim * (p->n_heads * head_size); // wq
    total_weights += n_layers * p->dim * (p->n_kv_heads * head_size); // wk
    total_weights += n_layers * p->dim * (p->n_kv_heads * head_size); // wv
    total_weights += n_layers * (p->n_heads * head_size) * p->dim; // wo
    total_weights += n_layers * p->dim; // rms_ffn_weight
    total_weights += n_layers * p->dim * p->hidden_dim; // w1
    total_weights += n_layers * p->hidden_dim * p->dim; // w2
    total_weights += n_layers * p->dim * p->hidden_dim; // w3
    total_weights += p->dim; // rms_final_weight
    if (!shared_weights) {
        total_weights += p->vocab_size * p->dim; // wcls
    }
    
    elem_t* elem_weights = (elem_t*)malloc(total_weights * sizeof(elem_t));
    if (!elem_weights) {
        fprintf(stderr, "Failed to allocate memory for elem_t weights\n");
        exit(EXIT_FAILURE);
    }
    
    // Convert all weights to elem_t
    float_to_elem(ptr, elem_weights, total_weights);
    
    // Assign pointers
    elem_t* elem_ptr = elem_weights;
    w->token_embedding_table = elem_ptr;
    elem_ptr += p->vocab_size * p->dim;
    w->rms_att_weight = elem_ptr;
    elem_ptr += n_layers * p->dim;
    w->wq = elem_ptr;
    elem_ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = elem_ptr;
    elem_ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = elem_ptr;
    elem_ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = elem_ptr;
    elem_ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = elem_ptr;
    elem_ptr += n_layers * p->dim;
    w->w1 = elem_ptr;
    elem_ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = elem_ptr;
    elem_ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = elem_ptr;
    elem_ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = elem_ptr;
    elem_ptr += p->dim;
    
    // Skip freq_cis_real and freq_cis_imag in original float data
    ptr += p->seq_len * head_size / 2;
    ptr += p->seq_len * head_size / 2;
    
    w->wcls = shared_weights ? w->token_embedding_table : elem_ptr;
}

// ----------------------------------------------------------------------------
// Gemmini-accelerated neural net blocks

void gemmini_rmsnorm(elem_t* o, elem_t* x, elem_t* weight, int size) {
    // Use Gemmini's norm function
    static acc_t norm_acc[DIM];
    
    tiled_norm_auto(1, size,
        (acc_t*)x, (elem_t*)o,
        ACC_SCALE_IDENTITY,
        LAYERNORM, WS);
    
    // Apply weight scaling
    for (int i = 0; i < size; i++) {
        o[i] = (elem_t)((float)o[i] * (float)weight[i] / ELEM_SCALE);
    }
    
    gemmini_fence();
}

void gemmini_softmax(elem_t* x, int size) {
    // For now, use CPU softmax - can be optimized later
    float max_val = -INFINITY;
    float* temp = (float*)malloc(size * sizeof(float));
    
    // Convert to float and find max
    for (int i = 0; i < size; i++) {
        temp[i] = (float)x[i] / ELEM_SCALE;
        if (temp[i] > max_val) max_val = temp[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        temp[i] = expf(temp[i] - max_val);
        sum += temp[i];
    }
    
    // Normalize and convert back
    for (int i = 0; i < size; i++) {
        x[i] = (elem_t)((temp[i] / sum) * ELEM_SCALE);
    }
    
    free(temp);
}

void gemmini_matmul(acc_t* xout, elem_t* x, elem_t* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // Using Gemmini's tiled matmul
    tiled_matmul_auto(1, d, n,
        /*A=*/ x, /*B=*/ w,
        /*D=*/ NULL, /*C=*/ xout,
        /*stride_A=*/n, /*stride_B=*/n, /*stride_D=*/0, /*stride_C=*/d,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        true, false,
        0,
        WS);
    
    gemmini_fence();
}

// ----------------------------------------------------------------------------
// Forward pass with Gemmini acceleration

elem_t* forward(Transformer* transformer, int token, int pos) {
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    elem_t *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    elem_t* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(elem_t));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        gemmini_rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position using Gemmini
        gemmini_matmul(s->x_acc, s->xb, w->wq + l*dim*dim, dim, dim);
        for (int i = 0; i < dim; i++) {
            s->q[i] = (elem_t)(s->x_acc[i] / ACC_SCALE_IDENTITY);
        }
        
        gemmini_matmul(s->x_acc, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        for (int i = 0; i < kv_dim; i++) {
            s->k[i] = (elem_t)(s->x_acc[i] / ACC_SCALE_IDENTITY);
        }
        
        gemmini_matmul(s->x_acc, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        for (int i = 0; i < kv_dim; i++) {
            s->v[i] = (elem_t)(s->x_acc[i] / ACC_SCALE_IDENTITY);
        }

        // RoPE relative positional encoding
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                elem_t* vec = v == 0 ? s->q : s->k;
                float v0 = (float)vec[i] / ELEM_SCALE;
                float v1 = (float)vec[i+1] / ELEM_SCALE;
                vec[i]   = (elem_t)((v0 * fcr - v1 * fci) * ELEM_SCALE);
                vec[i+1] = (elem_t)((v0 * fci + v1 * fcr) * ELEM_SCALE);
            }
        }

        // multihead attention
        for (int h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            elem_t* q = s->q + h * head_size;
            // attention scores for this head
            elem_t* att = s->att + h * p->seq_len;
            
            // compute attention scores
            for (int t = 0; t <= pos; t++) {
                elem_t* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                
                // dot product using Gemmini
                acc_t score_acc = 0;
                for (int i = 0; i < head_size; i++) {
                    score_acc += (acc_t)q[i] * (acc_t)k[i];
                }
                float score = (float)score_acc / (ELEM_SCALE * ELEM_SCALE);
                score /= sqrtf(head_size);
                att[t] = (elem_t)(score * ELEM_SCALE);
            }

            // softmax the scores
            gemmini_softmax(att, pos + 1);

            // weighted sum of the values
            elem_t* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(elem_t));
            
            for (int t = 0; t <= pos; t++) {
                elem_t* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                elem_t a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += (elem_t)(((acc_t)a * (acc_t)v[i]) / ACC_SCALE_IDENTITY);
                }
            }
        }

        // final matmul to get the output of the attention
        gemmini_matmul(s->xb2_acc, s->xb, w->wo + l*dim*dim, dim, dim);
        for (int i = 0; i < dim; i++) {
            s->xb2[i] = (elem_t)(s->xb2_acc[i] / ACC_SCALE_IDENTITY);
        }

        // residual connection back into x
        tiled_resadd_auto(1, dim,
            MVIN_SCALE_IDENTITY,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            x,
            s->xb2,
            x,
            /*relu=*/ false,
            WS);
        gemmini_fence();

        // ffn rmsnorm
        gemmini_rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // FFN: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // Calculate w1 and w3 projections
        gemmini_matmul(s->hb_acc, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = (elem_t)(s->hb_acc[i] / ACC_SCALE_IDENTITY);
        }
        
        gemmini_matmul(s->hb_acc, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            s->hb2[i] = (elem_t)(s->hb_acc[i] / ACC_SCALE_IDENTITY);
        }

        // SwiGLU activation (done on CPU for now)
        for (int i = 0; i < hidden_dim; i++) {
            float val = (float)s->hb[i] / ELEM_SCALE;
            val *= (1.0f / (1.0f + expf(-val)));
            val *= (float)s->hb2[i] / ELEM_SCALE;
            s->hb[i] = (elem_t)(val * ELEM_SCALE);
        }

        // final FFN matmul
        gemmini_matmul(s->xb_acc, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) {
            s->xb[i] = (elem_t)(s->xb_acc[i] / ACC_SCALE_IDENTITY);
        }

        // residual connection
        tiled_resadd_auto(1, dim,
            MVIN_SCALE_IDENTITY,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            x,
            s->xb,
            x,
            /*relu=*/ false,
            WS);
        gemmini_fence();
    }

    // final rmsnorm
    gemmini_rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    gemmini_matmul(s->logits_acc, x, w->wcls, p->dim, p->vocab_size);
    for (int i = 0; i < p->vocab_size; i++) {
        s->logits[i] = (elem_t)(s->logits_acc[i] / ACC_SCALE_IDENTITY);
    }
    
    return s->logits;
}

// ----------------------------------------------------------------------------
// The rest of the code (tokenizer, sampler, etc.) remains mostly the same
// Just need to handle elem_t to float conversions where necessary

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// Define ELEM_SCALE based on your Gemmini configuration
#ifndef ELEM_SCALE
#define ELEM_SCALE 127.0f
#endif

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // Temporary buffer for float logits
    float* float_logits = (float*)malloc(transformer->config.vocab_size * sizeof(float));

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        elem_t* logits = forward(transformer, token, pos);
        
        // Convert elem_t logits to float for sampling
        elem_to_float(logits, float_logits, transformer->config.vocab_size);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, float_logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(float_logits);
    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
    }
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // Temporary buffer for float logits
    float* float_logits = (float*)malloc(transformer->config.vocab_size * sizeof(float));

    int8_t user_turn = 1;
    int next;
    int token;
    int prev_token;
    int pos = 0;
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == NULL) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            if (pos == 0 && cli_user_prompt != NULL) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) { user_turn = 1; }

        elem_t* logits = forward(transformer, token, pos);
        elem_to_float(logits, float_logits, transformer->config.vocab_size);
        next = sample(sampler, float_logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(float_logits);
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   gemmini_run <checkpoint> [options]\n");
    fprintf(stderr, "Example: gemmini_run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // Initialize Gemmini
    gemmini_flush(0);

    // default parameters
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;

    // parse arguments
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
