# Releasing

1. Run every command in `CONTRIBUTING.md` and a live `start/status/stop` smoke.
2. Review `config/models.lock.json`; unknown source or licence fields block any
   redistribution of the corresponding model, but not a code-only release.
3. Move Unreleased changelog entries into a dated semantic version section.
4. Update `system.version` in `config/system.toml` and create an annotated tag.
5. Push the branch and tag, then attach only source artifacts. Never attach
   `.env`, `.kindred/`, logs, local media, model weights, or nested checkouts.

GitHub branch protection and private vulnerability reporting are repository
settings and must be verified by an administrator after the first push.
