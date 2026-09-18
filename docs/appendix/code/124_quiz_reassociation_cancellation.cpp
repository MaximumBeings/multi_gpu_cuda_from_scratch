// Appendix B: Practice Quiz
// 124_quiz_reassociation_cancellation.cpp
//
// Appendix B.4, Challenge 2 -- Chapters 25, 29, and 31 each independently
// rediscovered that floating-point addition is not associative: summing
// the SAME set of values in a different ORDER (a real, unavoidable
// consequence of which rank's partial result a reduction combines first)
// can produce a genuinely different final bit pattern, not just a
// hypothetical rounding worry. Before compiling and running this file,
// predict: given the four doubles a=1e16, b=1.0, c=-1e16, d=1.0, does
// the LEFT-TO-RIGHT sequential sum ((a+b)+c)+d equal the PAIRWISE sum
// (a+c)+(b+d)? Both orders add the exact same four numbers.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 124_quiz_reassociation_cancellation.cpp -o 124_quiz_reassociation_cancellation
// Run:     ./124_quiz_reassociation_cancellation
#include <cstdio>

int main() {
    double a = 1e16, b = 1.0, c = -1e16, d = 1.0;

    printf("=== summing the same four doubles in two different orders ===\n");
    printf("a=%.1f, b=%.1f, c=%.1f, d=%.1f\n\n", a, b, c, d);

    // Sequential, left-to-right (the order Chapter 25's naive baseline
    // combines ring-forwarded partial sums in).
    double step1 = a + b;
    double step2 = step1 + c;
    double sequential = step2 + d;
    printf("sequential ((a+b)+c)+d:\n");
    printf("  a+b       = %.1f  (b=1.0 is far smaller than a's own ULP at "
           "this magnitude -- it vanishes in the rounding)\n", step1);
    printf("  (a+b)+c   = %.1f\n", step2);
    printf("  ((a+b)+c)+d = %.1f\n\n", sequential);

    // Pairwise, the order Chapter 31's tree-style combine groups values in.
    double left = a + c;
    double right = b + d;
    double pairwise = left + right;
    printf("pairwise (a+c)+(b+d):\n");
    printf("  a+c       = %.1f  (equal magnitude, opposite sign -- exact "
           "cancellation, no rounding loss)\n", left);
    printf("  b+d       = %.1f\n", right);
    printf("  (a+c)+(b+d) = %.1f\n\n", pairwise);

    printf("sequential result = %.1f\n", sequential);
    printf("pairwise result   = %.1f\n", pairwise);
    printf("difference        = %.1f\n\n", pairwise - sequential);

    bool match = (sequential == pairwise);
    printf("self-check: do the two orders agree? %s -- both orders sum "
           "the IDENTICAL four numbers, so any difference is entirely a "
           "property of ROUND-OFF ORDER, not of the data itself. This is "
           "the exact same real mechanism Chapters 25/29/31 each found: "
           "a ring's own sequential forwarding order and a tree-style "
           "pairwise combine order are NOT required to produce bit-"
           "identical sums, even though both are mathematically \"the "
           "same\" reduction.\n",
           match ? "YES (unexpected for these values)" : "NO");
    return 0;
}
