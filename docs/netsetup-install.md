# netsetup one-time install (sudo, Ubuntu 22.04)

The runtime runs headless/SSH and must manage NetworkManager profiles without
sudo. Ubuntu 22.04 ships **polkitd 0.105**, which only reads LocalAuthority
`.pkla` files — JavaScript `.rules` files are silently ignored (the binary has
no JS engine; do not cargo-cult Arch/Fedora advice).

## 1. polkit grant

Create `/etc/polkit-1/localauthority/50-local.d/46-apollo-networkmanager.pkla`
with exactly:

```ini
[apollo: let netdev group manage NetworkManager]
Identity=unix-group:netdev
Action=org.freedesktop.NetworkManager.network-control;org.freedesktop.NetworkManager.settings.modify.system
ResultAny=yes
ResultInactive=yes
ResultActive=yes
```

```bash
sudo tee /etc/polkit-1/localauthority/50-local.d/46-apollo-networkmanager.pkla <<'EOF'
[apollo: let netdev group manage NetworkManager]
Identity=unix-group:netdev
Action=org.freedesktop.NetworkManager.network-control;org.freedesktop.NetworkManager.settings.modify.system
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOF
```

polkitd watches the directory — no restart needed.

## 2. netdev group

```bash
sudo usermod -aG netdev $USER   # group exists on Ubuntu (gid 120); RE-LOGIN required
```

## 3. Verify

```bash
uv run python -m apollo_mavis_v2_hardware.netsetup install --check
```

`verify()` also runs this check at every session start, so a missing grant
shows up as an actionable landing-page warning instead of a mid-bring-up
failure. The guided path (prints these commands, asks confirmation, runs them
via sudo) is:

```bash
uv run python -m apollo_mavis_v2_hardware.netsetup install
```

## Notes

- Probing needs no privileges at all: `ping` works via `ping_group_range`,
  ARP is cross-checked with `ip neigh show` (not `arping`), and the port-502
  reachability check is an ordinary TCP connect (nothing is written — 502 is
  the live SDK control channel).
- The tool never touches the internet NIC: devices on the lowest-metric
  default route and every non-ethernet device are hard-denylisted from all
  mutating paths.
- `emergency_stop()` in the driver is a software stop (`set_state(4)`), not a
  hardware STO, and it does not clear errors — the physical e-stop button on
  the control box remains the real emergency path.
- For Ubuntu 24.04+ (polkit >= 123) the same grant moves to a JS rule in
  `/etc/polkit-1/rules.d/`.
