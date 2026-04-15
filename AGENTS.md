## Approach

- Read before editing. Test before declaring done.
- Prefer small edits over rewrites.
- Reproduce before fixing issues that depend on runtime behavior or external services.
- Unproven concerns are risks, not bugs. Say so if you haven't reproduced it.
- Simplest working solution. No over-engineering, no speculative features, no abstractions for single-use ops.

## Output

- Code first. Explanation and comments only when logic is non-obvious.
- No filler, no boilerplate, no suggestions outside scope.

## Code

- Remove unused imports, variables, parameters, dead branches, dead functions in files you edit.
- No error handling for impossible scenarios.
- All imports at top of file. No imports inside functions unless strictly required to break circular dependencies.
- Code and comments in English. User-facing strings stay in their original language.

## Debugging

- Read code before explaining. Prove with direct evidence: failing test, reproduced run, or concrete probe.
- State what you found, where, and the fix. If unclear, say so.

## Verification

- Smallest proof first, then broader checks.
- Use the language's standard toolchain. Default checks: format, lint (warnings as errors), tests. Skip only with stated reason.
- No "fixed/safe/ready" claims without fresh command output.

## Git

- Ask before pushing every time, even if previously approved. No batch commit+push. No force push or hard reset without approval.
- Merge to `main` with a single squashed commit. Commit messages in English.

## Formatting

- Plain hyphens and straight quotes only. No decorative Unicode. Code output copy-paste safe.
