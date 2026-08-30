# Compatibility Contracts

Files in this directory are language-neutral behavioral fixtures. During the
hybrid migration, both Python and Rust tests read the same fixture and must
produce the same expected result.

Contracts contain only synthetic, non-identifying values. They describe public
behavior and stored-data semantics, not source layout or implementation details.
