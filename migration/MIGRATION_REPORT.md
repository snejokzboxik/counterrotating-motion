# V17C migration report

The 14 historical V17C branch refs and their exact commits have been published at `snejokzboxik/counterrotating-motion`. The new `main` descends from V17C.22 and contains the V17C catalog, recovered local files, and migration audit. The historical branches are unchanged.

A source mirror contained the full commit graph and all Git objects; `git fsck --full --strict` passed before publication. Remote branch SHA comparison, clean clone, all scientific file SHA-256 checks, selected tests/data loads, TeX availability, and independent bundle verification are documented in the manifest and file integrity report.

The source master had no `research/v17c/` catalog and was not changed. The source `research/v17b` branch also lacks that catalog. Source branch deletion is recorded separately after the final pre-deletion SHA check.
