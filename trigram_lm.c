
// trigram_lm.c — modello a trigrammi (carattere-level) in C
// Build: gcc -O3 -std=c11 -Wall -Wextra -o trigram_lm trigram_lm.c
// Uso: ./trigram_lm <corpus.txt> [output_len] [seed]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define V 256   // byte-charset

static unsigned char *corpus = NULL;
static size_t corpus_sz = 0;

// counts[a][b][c] = quante volte (a,b)->c appare
static uint32_t ***counts;

// alloca matrice 3D
static void alloc_counts() {
    counts = malloc(V * sizeof(uint32_t **));
    for (int a = 0; a < V; a++) {
        counts[a] = malloc(V * sizeof(uint32_t *));
        for (int b = 0; b < V; b++) {
            counts[a][b] = malloc(V * sizeof(uint32_t));
            for (int c = 0; c < V; c++) {
                counts[a][b][c] = 1u; // Laplace smoothing
            }
        }
    }
}

static void die(const char *m) {
    fprintf(stderr, "%s\n", m);
    exit(1);
}

// carica file intero
static void load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("Cannot open file");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) die("ftell failed");
    fseek(f, 0, SEEK_SET);
    corpus_sz = (size_t)sz;
    corpus = malloc(corpus_sz);
    fread(corpus, 1, corpus_sz, f);
    fclose(f);
    if (!corpus_sz) die("Corpus empty");
}

// training: trigram (prev2, prev1) -> cur
static void train_trigram() {
    if (corpus_sz < 3) die("Corpus too small");
    unsigned char p2 = corpus[0];
    unsigned char p1 = corpus[1];
    for (size_t i = 2; i < corpus_sz; i++) {
        unsigned char c = corpus[i];
        counts[p2][p1][c]++;
        p2 = p1;
        p1 = c;
    }
}

// sampling da counts[p2][p1][*]
static unsigned char sample_next(unsigned char p2, unsigned char p1) {
    uint64_t total = 0;
    for (int c = 0; c < V; c++) total += counts[p2][p1][c];

    uint64_t r = ((uint64_t)rand() << 32 ^ rand()) % total;

    uint64_t acc = 0;
    for (int c = 0; c < V; c++) {
        acc += counts[p2][p1][c];
        if (acc > r) return (unsigned char)c;
    }
    return 0;
}

// filtro ASCII leggibile (come il tuo bigram)
static unsigned char printable(unsigned char c) {
    if (c == '\n') return c;
    if (c >= 32 && c <= 126) return c;
    return ' ';
}

int main(int argc, char **argv) {
    if (argc < 2) die("Usage: ./trigram_lm <corpus> [out_len] [seed]");

    size_t out_len = (argc >= 3 ? strtoul(argv[2], NULL, 10) : 500);
    unsigned seed  = (argc >= 4 ? strtoul(argv[3], NULL, 10) : time(NULL));
    srand(seed);

    load(argv[1]);
    alloc_counts();
    train_trigram();

    // seed iniziali = ultimi due byte del corpus
    unsigned char p2 = corpus[corpus_sz - 2];
    unsigned char p1 = corpus[corpus_sz - 1];

    for (size_t i = 0; i < out_len; i++) {
        unsigned char c = sample_next(p2, p1);
        c = printable(c);
        putchar(c);
        p2 = p1;
        p1 = c;
    }

    return 0;
}

