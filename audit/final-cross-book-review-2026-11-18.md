# Final cross-book adversarial review — November 18, 2026

## Multi-GPU findings corrected

- IPC shares an allocation; remote reads still move data. It does not imply zero physical traffic.
- Equal-slot pipeline concurrency can survive a slot barrier; global waiting can still obstruct asynchronous progress. Activation memory depends on size, live buffers and checkpoint policy, not only microbatch count.
- cuBLAS-Xt supports host matrices and matrices on participating devices. Exercise answers now match the numerical question, and an earlier cleanup diagnostic is not presented as proof of a universal current library defect.
- All-gather is one exact N-body implementation; streaming/ring exchanges can provide all sources without retaining the entire global dataset on each device. Exact quantiles also need not gather every value.
- Explanatory ASCII panels no longer equate every collective with reduction, potential split-node counts with actual collective invocations, or independent synthetic records with an entire genomics workflow. Nine text diagrams were corrected and aligned.
- Stale setup/compiler commentary was corrected. Existing worked examples retain their declared numeric/data-ownership contracts.
- All 47 worked illustrations now have full-size SVG links; 33 previously used unlinked images. The figures continue to fit the text column. All 195 SVGs passed embedded-glyph review; the two existing validators cover the 148 original and 47 worked assets respectively.

## Method and limits

Three agents independently reviewed H100 content, multi-GPU content and visual/worked material, followed by parent integration. Content coverage includes the chapter sequence, setup, worked examples, summaries and exercises. Visual checks combine automated inventory/font checks with numerical worked-figure checks, contact sheets and targeted full-size renders; this is not a claim that every legacy figure was manually inspected at full size.

Tested executable sources, downloadable source revisions, raw logs, captured commands and execution records are unchanged. Incorrect non-executable explanatory text diagrams were corrected rather than retained as contradictory teaching material. No GPU host was accessed and no new GPU measurement is implied. Existing source/evidence consistency checks remain authoritative for the precise tested revisions and scope. No new GPU numerical defect was identified during this review.

Installation shell blocks were parsed with Bash without execution. NVIDIA driver, CUDA 12.8 Update 1, NCCL and Fabric Manager documentation were rechecked; the setup guides now distinguish the selected corresponding-driver baseline from the more limited CUDA minor-version compatibility path. Installation on a newly provisioned host was not repeated.

Validation includes the repository's execution-evidence checks, strict MkDocs build, built internal links/anchors, displayed-source preservation and SVG checks. Source commits and Pages deployments are checked after pushing; canonical published articles and SVG/CSS hashes are compared with the final local build.

## Primary references

- [CUDA programming guide, clusters and memory model](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html).
- [CUDA runtime synchronization behavior](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html).
- [CUDA 12.8 Update 1 driver compatibility tables](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-driver).
- [NVIDIA Ubuntu driver installation](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/ubuntu.html).
- [cuBLAS-Xt API](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublasxt-api).
- [GPipe](https://arxiv.org/abs/1811.06965).

## Local verification result

Local results: 54 HTML pages; 33,821 internal targets/fragments; all 145 displayed source listings and execution-evidence checks passed. The nine changed explanatory fences and their before/after hashes are recorded in `final-explanatory-fence-changes-2026-11-18.json`.
