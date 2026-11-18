// Chapter 30: Multi-GPU Ray Tracing and Rendering
// 87_independent_tile_rendering_correctness_simulation.cpp
//
// Every case study since Chapter 24 needed SOME communication DURING
// computation: Chapter 16/29's halo exchange every step, Chapter 9/25's
// ring all-reduce, Chapter 26's tensor/pipeline-parallel rounds, Chapter
// 27's mandatory all-gather, Chapter 28's graph-dependent frontier
// exchange, Chapter 29's residual Allreduce. Rendering an image is
// structurally different in the simple case: Wikipedia's own summary of
// how real render farms work states it plainly -- "frames and sometimes
// tiles can be calculated independently of the others, with the main
// communication between processors being the upload of the initial
// source material, such as models and textures, and the download of the
// finished images." Molnar, Cox, Ellsworth & Fuchs's own real classic
// taxonomy ("A Sorting Classification of Parallel Rendering," IEEE
// Computer Graphics and Applications, 1994) calls this shape "sort-last":
// sorting (assembling results into the final image) happens only "after
// primitives have been rasterized into pixels, samples, or pixel
// fragments." This file builds the simplest possible version of that
// idea: a tiny fixed 2D "scene" (two circles standing in for spheres,
// nearest-hit-wins, matching real ray tracing's depth test), a
// deterministic integer-only per-pixel hit test with NO floating point
// and NO shared mutable state, split into row-tiles across P ranks. Each
// rank computes its OWN tile using ONLY the fixed scene description and
// its own pixel coordinates -- it never reads another rank's tile, and
// there is no reduction step at all (unlike every earlier chapter's
// collective), so there is not even an associativity question to ask.
// The assembled image is checked against a single-process reference,
// exactly, across several values of P.
#include <cstdio>
#include <vector>

const int WIDTH = 16;
const int HEIGHT = 12;

struct Circle { int cx, cy, r2; int color; }; // r2 = radius squared

// A fixed, deterministic 2-object "scene." Circle 0 is drawn in front of
// (i.e., wins ties/overlaps against) circle 1 when both are hit, exactly
// like a real ray tracer's nearest-hit rule -- here, "nearest" is modeled
// as "smaller squared distance to the object's own center," a simplified
// stand-in for real depth comparison.
std::vector<Circle> makeScene() {
    return {
        {5, 5, 9, 1},   // color 1: circle at (5,5), radius^2 = 9 (r=3)
        {11, 7, 16, 2}  // color 2: circle at (11,7), radius^2 = 16 (r=4)
    };
}

// Pure per-pixel function: given ONLY the fixed scene and this pixel's
// own (x,y), returns the pixel's color. No other pixel's data, and no
// other rank's data, is ever touched.
int shadePixel(const std::vector<Circle> &scene, int x, int y) {
    int bestColor = 0;      // 0 = background
    int bestDistSq = -1;    // -1 = "no hit yet"
    for (const Circle &c : scene) {
        int dx = x - c.cx, dy = y - c.cy;
        int distSq = dx * dx + dy * dy;
        if (distSq <= c.r2) {
            if (bestDistSq == -1 || distSq < bestDistSq) {
                bestDistSq = distSq;
                bestColor = c.color;
            }
        }
    }
    return bestColor;
}

// --- Reference: single-process render of the FULL image. ---
std::vector<int> referenceRender() {
    auto scene = makeScene();
    std::vector<int> image(WIDTH * HEIGHT);
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++)
            image[y * WIDTH + x] = shadePixel(scene, x, y);
    return image;
}

// --- Distributed: P ranks, each rendering its own row-tile, with ZERO
// communication during rendering -- no halo, no reduction, nothing. ---
std::vector<int> distributedRender(int P) {
    auto scene = makeScene();
    int rowsPerRank = HEIGHT / P;
    std::vector<int> assembled(WIDTH * HEIGHT, -999); // sentinel: "not yet written"

    for (int r = 0; r < P; r++) {
        int yStart = r * rowsPerRank;
        int yEnd = yStart + rowsPerRank;
        // This rank touches ONLY its own rows, using ONLY the fixed scene
        // and its own pixel coordinates -- no other rank's tile is read.
        for (int y = yStart; y < yEnd; y++)
            for (int x = 0; x < WIDTH; x++)
                assembled[y * WIDTH + x] = shadePixel(scene, x, y);
    }
    return assembled;
}

void printImage(const std::vector<int> &image) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int c = image[y * WIDTH + x];
            putchar(c == 0 ? '.' : (c == 1 ? '1' : '2'));
        }
        putchar('\n');
    }
}

int main() {
    std::vector<int> ref = referenceRender();

    printf("Reference (single-process) %dx%d render, 2-circle scene "
           "('.'=background, '1'/'2'=circle color, nearest hit wins):\n\n", WIDTH, HEIGHT);
    printImage(ref);
    printf("\n");

    int Ps[] = {1, 2, 3, 4, 6, 12};
    printf("%-6s %-22s %-30s\n", "P", "rows/rank", "assembled image EXACT match?");
    for (int P : Ps) {
        std::vector<int> dist = distributedRender(P);
        bool exact = (dist == ref);
        printf("%-6d %-22d %-30s\n", P, HEIGHT / P, exact ? "YES" : "NO");
    }

    return 0;
}
