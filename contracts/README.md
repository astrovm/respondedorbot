# Compatibility Contracts

Files in this directory are language-neutral behavioral and persistence
fixtures. Rust tests use them to preserve the compatibility baseline established
before the native cutover.

Contracts contain only synthetic, non-identifying values. They describe public
behavior and stored-data semantics, not source layout or implementation details.
