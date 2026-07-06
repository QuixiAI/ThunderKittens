# Security Policy

QuixiCore CUDA is a native GPU backend. Security issues may involve host-side
bindings, kernel launch validation, memory bounds, build tooling, or packaged
artifacts.

## Reporting

Do not open a public issue for a suspected vulnerability. Report security
issues through the QuixiAI GitHub security advisory flow or contact the
maintainers privately.

When reporting, include:

- Affected repository and commit.
- Affected hardware and driver/runtime versions.
- Minimal reproduction steps.
- Whether the issue affects public APIs, bindings, generated artifacts, or
  kernel execution.

## Scope

Issues in shared QuixiCore semantics should be reported against
QuixiAI/QuixiCore. Issues in CUDA implementation code should be reported
against this repository.
