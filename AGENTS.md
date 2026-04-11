## Approach

- Read before editing. Test before declaring done.
- Prefer small edits over rewrites.
- Probe before fixing issues that depend on runtime behavior or external services.
- Unproven concerns are risks, not bugs. Say so if you haven't reproduced it.
- Simple, direct solutions. User instructions override this file.

## Output

- Code first. Explanation only if non-obvious.
- Comments only when logic is not obvious. No boilerplate.

## Code

- Simplest working solution. No over-engineering, no speculative features.
- No abstractions for single-use operations. Three similar lines beats a premature abstraction.
- When touching code: remove unused imports, variables, parameters, dead branches, and dead functions.
- Simplify changed code to the smallest clear version that preserves behavior.
- No error handling for impossible scenarios.
- New imports go at the top of the file.

## Review

- State the bug. Show the fix. Stop.
- No suggestions outside scope. No filler.

## Debugging

- Read the code before explaining the bug.
- Prove with direct evidence: failing test, reproduced run, or concrete probe.
- State what you found, where, and the fix. If unclear, say so.

## Verification

- Use the project's actual tools.
- Run the smallest proof first, then broader checks for the touched area.
- Default checks: format, lint (warnings as errors), tests. Skip only with stated reason.
- No "fixed/safe/ready" claims without fresh command output.

## Git

- Merge to `main` with a single squashed commit.
- Commit messages must always be in English.

## Formatting

- Plain hyphens and straight quotes only. No decorative Unicode.
- Code output must be copy-paste safe.
