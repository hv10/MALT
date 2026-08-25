## DONE

- [x] finish refactoring into exchangeable components
- [x] add one more modality as proof-of-concept
- [x] add HiPPO.jl as embedder
- [x] make a build system using a `.toml` file
- [x] support for tabular data
- [x] add custom cmd as embedder
- [x] better CLI setup with `malt` and `malt tool` commands
- added tools:
  - [x] soft-label assignment for image-patches
  - [x] provide (`uvx`-callable) tooling for splitting images into overlapping patches

## Enhancements
- add a bunch more modalities
  - [x] text
  - [ ] genome data?
- [ ] introduce "lenses"
  - [ ] allow more than one embedding per sample
  - [ ] allow switching between sample embeddings in UI
  - [ ] allow transformation of sample embeddings with "Kerneltrick"
- [ ] introduce new task "alignment" with the goal of finding the transport function between two embedding spaces based on labels
  - [ ] introduce tool to convert soft-labeling to a transport function between two embedding spaces
- [ ] introduce alternate classifier models aside from PAClf
  - [ ] relax requirements of classifiers (margin yes, but arbitrary decision boundary)
- [ ] make a build script that _makes_ the single-file version in compact form based on `build.toml` by combining code

## Non-code additions
- [ ] define use-case example for post-hoc analysis of embedding spaces
- [ ] document the system, including the extensibility interfaces
- [ ] document design choices
- [ ] provide tooling for creating soft-assignment label maps for split image annotations

## Fixes
- [ ] make a truly online mode (i.e. PAClf with partial_fit)

## QoL improvements
- [ ] see if we want / can outsource some functionality to Julia via Juliacall and if it will make the projections faster