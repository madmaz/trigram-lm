
// trigram_lm_plus.c — trigram LM potenziato con temperatura e fallback
// Build: gcc -O3 -std=c11 -Wall -Wextra -o trigram_lm_plus trigram_lm_plus.c
// Usage: ./trigram_lm_plus corpus.txt 1000 42 1.0
//                                     ^len ^seed ^temp

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define V 256

// --- GLOBALS ---
static unsigned char *corpus = NULL;
static size_t corpus_sz = 0;

// counts trigram: counts[a][b][c]
static uint32_t ***tri;
// bigram: counts[b][c]
static uint32_t **bi;
// unigram: counts[c]
static uint32_t *uni;

// --- UTILS ---
static void die(const char *m) {
    fprintf(stderr, "%s\n", m);
    exit(1);
}

static void load_file(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) die("Cannot open file");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) die("Empty or invalid file");
    fseek(f, 0, SEEK_SET);
    corpus_sz = (size_t)sz;
    corpus = malloc(corpus_sz);
    fread(corpus, 1, corpus_sz, f);
    fclose(f);
}

// --- ALLOC ---
static void alloc_counts() {
    tri = malloc(V * sizeof(uint32_t **));
    bi  = malloc(V * sizeof(uint32_t *));
    uni = malloc(V * sizeof(uint32_t));

    for (int a = 0; a < V; a++) {
        tri[a] = malloc(V * sizeof(uint32_t *));
        bi[a]  = malloc(V * sizeof(uint32_t));
        for (int b = 0; b < V; b++) {
            tri[a][b] = malloc(V * sizeof(uint32_t));
            bi[a][b] = 1u;  // smoothing
            for (int c = 0; c < V; c++) {
                tri[a][b][c] = 1u; // smoothing
            }
        }
    }
    for (int c = 0; c < V; c++) uni[c] = 1u;
}

// --- TRAIN ---
static void train_model() {
    if (corpus_sz < 3) die("Corpus too small");

    unsigned char p2 = corpus[0];
    unsigned char p1 = corpus[1];

    uni[p2]++;
    uni[p1]++;
    bi[p2][p1]++;

    for (size_t i = 2; i < corpus_sz; i++) {
        unsigned char c = corpus[i];

        tri[p2][p1][c]++;
        bi[p1][c]++;
        uni[c]++;

        p2 = p1;
        p1 = c;
    }
}

// --- SAMPLING WITH TEMPERATURE ---
static unsigned char sample_from_dist(double *prob, double temp) {
    double total = 0.0;
    for (int c = 0; c < V; c++) total += pow(prob[c], 1.0 / temp);

    double r = ((double)rand() / RAND_MAX) * total;
    double acc = 0.0;

    for (int c = 0; c < V; c++) {
        acc += pow(prob[c], 1.0 / temp);
        if (acc >= r) return (unsigned char)c;
    }
    return 0;
}

// costruisce distribuzione normalizzata
static unsigned char sample_next(unsigned char p2, unsigned char p1, double temp) {
    double prob[V];

    // 1) trigram
    double sum = 0.0;
    for (int c = 0; c < V; c++) sum += tri[p2][p1][c];
    if (sum > V * 1.0001) { // se davvero ha info
        for (int c = 0; c < V; c++)
            prob[c] = tri[p2][p1][c] / sum;
        return sample_from_dist(prob, temp);
    }

    // 2) fallback bigram
    sum = 0.0;
    for (int c = 0; c < V; c++) sum += bi[p1][c];
    if (sum > V * 1.0001) {
        for (int c = 0; c < V; c++)
            prob[c] = bi[p1][c] / sum;
        return sample_from_dist(prob, temp);
    }

    // 3) fallback unigram
    sum = 0.0;
    for (int c = 0; c < V; c++) sum += uni[c];
    for (int c = 0; c < V; c++)
        prob[c] = uni[c] / sum;

    return sample_from_dist(prob, temp);
}

// filter character
static unsigned char printable(unsigned char c) {
    if (c == '\n') return c;
    if (c >= 32 && c <= 126) return c;
    return ' ';
}

// --- MAIN ---
int main(int argc, char **argv) {
    if (argc < 2) die("Usage: ./trigram_lm_plus <corpus> [len] [seed] [temp]");

    size_t out_len = (argc >= 3 ? strtoul(argv[2], NULL, 10) : 500);
    unsigned seed  = (argc >= 4 ? strtoul(argv[3], NULL, 10) : time(NULL));
    double temp    = (argc >= 5 ? atof(argv[4]) : 1.0);

    srand(seed);

    load_file(argv[1]);
    alloc_counts();
    train_model();

    unsigned char p2 = corpus[corpus_sz - 2];
    unsigned char p1 = corpus[corpus_sz - 1];

    for (size_t i = 0; i < out_len; i++) {
        unsigned char c = sample_next(p2, p1, temp);
        c = printable(c);
        putchar(c);
        p2 = p1;
        p1 = c;
    }

    return 0;
}
