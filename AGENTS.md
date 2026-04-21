## Approach

- Read before editing. Test before declaring done.
- Prefer small edits over rewrites.
- Reproduce before fixing runtime or external issues.
- Unproven concerns are risks, not bugs. Say so if not reproduced.
- Simplest working solution. No over-engineering, speculative features, or single-use abstractions.

## Output

- Code first. Explain only non-obvious logic.
- No filler, boilerplate, or out-of-scope suggestions.

## Code

- Remove unused imports, variables, parameters, dead branches, and dead functions from edited files.
- No error handling for impossible scenarios.
- All imports at top of file. None inside functions unless strictly required to break circular dependencies.
- Code and comments in English. User-facing strings stay in their original language.

## Maintenance

- Remove old code when introducing replacements. No backward compatibility shims without explicit authorization.
- Do not preserve feature flags for shipped features or abstractions that serve a single caller.

## Debugging

- Read code before explaining. Prove with direct evidence: failing test, reproduced run, or concrete probe.
- State what you found, where, and the fix. If unclear, say so.

## Verification

- Smallest proof first, then broader checks.
- Use the standard toolchain. Default checks: format, lint (warnings as errors), tests. Skip only with stated reason.
- No "fixed/safe/ready" claims without fresh command output.
- Fix every issue you encounter. There are no pre-existing bugs or errors to ignore.

## Tests

- If the project has tests, run them before committing or declaring work complete. No exceptions.
- A failing test is a blocking issue. Fix it before moving on.

## Git

- Ask before pushing every time, even if previously approved.
- No batch commit+push. No force push or hard reset without approval.
- Merge to `main` with a single squashed commit. Commit messages in English.

## Configuration

- Environment variables only for secrets and external credentials.
- Prioritize sane defaults, zero-config, and easy maintenance. Hardcode sensible defaults for internal URLs, ports, and feature flags.
- When adding a dependency, verify the actual latest version from the registry or official source. Never rely on model memory.

## Formatting

- Plain hyphens and straight quotes only. No decorative Unicode. Code output copy-paste safe.
