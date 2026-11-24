# V17C file integrity and reproducibility

A fresh partial clone of the published target supplied all 3095 unique V17C Git blobs referenced by 28577 branch/file records. Their actual bytes (2,616,869,182 total unique bytes) matched the source SHA-256 ledger with zero mismatches. Ledger SHA-256: `95b9caca2165520e438ba8578b14b4e3881208099fadf4fad1840afcd4ccae7e`.

The 14 source/target branch SHAs match exactly. The 8 recovered local files also matched their recorded SHA-256 in the fresh clone. No Git LFS objects or GitHub release assets were found.

Selected published-state tests: 17 V14/V15 tests and 6 subtests, 4 V18B19 tests, and 9 V21A/B/V22 tests passed. V20B, V21, and V22 NPZ arrays loaded without pickle.

Tectonic compiled the standalone V21A and V22 analytical TeX documents from the fresh clone. Historical V14 English TeX fails at a literal backtick/underscore expression; V21 Russian TeX needs a T2A font not provided by this Tectonic installation; V21B is a LaTeX fragment without a preamble. These are source/environment issues, not missing migrated files. The historical originals remain unchanged on archive branches.

The full-history bundle passed `git bundle verify`; its SHA-256 is `50c1f0553043a2eae0a8d1409676c80107b370f1349cfbd6b4fb08406f566fb5`, size 4,979,797,730 bytes. The bundle is outside the source repository and its worktrees.
