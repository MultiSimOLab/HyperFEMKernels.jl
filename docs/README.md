# Documentation

Documenter.jl site, to be set up later.

# Releasing a version

The package is registered in the [General registry](https://github.com/JuliaRegistries/General)
with [Registrator](https://github.com/JuliaRegistries/Registrator.jl); GitHub releases are created by
[TagBot](https://github.com/JuliaRegistries/TagBot) (`.github/workflows/TagBot.yml`).

1. Bump `version` in `Project.toml` (semantic versioning: patch for fixes, minor for new features
   and, while in 0.x, for breaking changes) and merge into `main` through a pull request.
2. On GitHub, comment `@JuliaRegistrator register` on the merge commit of `main` (or on an issue,
   which registers the tip of `main`). Registrator opens a pull request in General.
3. The General PR merges automatically if the checks pass: every dependency has a `[compat]` entry
   with an upper bound, the version follows from the previous one, the name is acceptable.
   A **new package** waits three days before merging. Problems are reported as comments on that PR.
4. Once merged, TagBot tags `v<version>` and creates the GitHub release with the changelog.

Requirements on the repository: the JuliaRegistrator and TagBot GitHub apps installed
(organisation settings → GitHub Apps), the repository public, `LICENSE` in the root.
`CompatHelper.yml` opens pull requests when dependencies release versions outside `[compat]`;
Gridap is pinned on purpose (`=0.19.9`, HyperFEM's version), so its pull requests are closed until
HyperFEM upgrades.
