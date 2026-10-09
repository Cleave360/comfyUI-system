# Security Policy

## Supported version

Security fixes are applied to the current `main` branch. Releases are identified
by Git tags and `CHANGELOG.md`.

## Reporting

Do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability reporting for `Cleave360/comfyUI-system`, or contact the repository
owner privately if that feature is unavailable. Include reproduction steps,
affected revision, impact, and any proposed mitigation. Do not include API keys,
model credentials, `.env`, or `.kindred/` evidence in a report.

## Boundaries

- Services bind to loopback by default. Non-loopback Jazzy binding requires a
  WebSocket token and explicit origin allowlist.
- A Python virtual environment isolates dependencies; it is not an operating
  system sandbox and does not make third-party custom nodes trustworthy.
- Jazzy workflow dispatch fails closed unless Adaptive accepts a governed start
  event. This does not sandbox ComfyUI or prove a model/custom node safe.
- `.kindred/`, runtime logs, model weights, local media, and secrets must not be
  committed.
- Model source and licence fields marked unknown in `config/models.lock.json`
  require human review before redistribution.
