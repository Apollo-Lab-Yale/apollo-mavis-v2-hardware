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

## 3. NM dispatcher hook + system state (boot / hot-plug, any account)

Root re-runs `netsetup match --repair` on every ethernet `up`/`down` event, so
the arm NICs land on their profiles at boot and after cable (re)plugs/swaps even
when nobody is logged in (no polkit involved). Design: 02-hardware §7.4.

```bash
sudo install -d -m 755 -o root -g root /etc/apollo-mavis-v2
sudo install -D -m 755 -o root -g root <rendered script> /etc/NetworkManager/dispatcher.d/90-mavis-netsetup
# seed /etc/apollo-mavis-v2/nic_map.json + pin both profiles (arms must answer TCP 502):
sudo /path/to/.venv/bin/python -m apollo_mavis_v2_hardware.netsetup match \
    --state /etc/apollo-mavis-v2/nic_map.json --arm view=192.168.2.219 --arm grip=192.168.1.201
```

The script is rendered by `install` (venv python + arm list baked in); log:
`/var/log/mavis-netsetup.log`; lock: `/run/lock/mavis-netsetup.lock`.

## 4. Verify

```bash
uv run python -m apollo_mavis_v2_hardware.netsetup install --check \
    --arm view=192.168.2.219 --arm grip=192.168.1.201
```

`verify()` also runs this check at every session start, so a missing grant or
hook shows up as an actionable landing-page warning instead of a mid-bring-up
failure. The guided path (prints every sudo command, asks confirmation, runs
them, then seeds the system nic_map) is:

```bash
uv run python -m apollo_mavis_v2_hardware.netsetup install \
    --arm view=192.168.2.219 --arm grip=192.168.1.201      # or --config <workcell.yaml>
# one-liner, non-interactive (SUDO_USER is the account added to netdev):
sudo /path/to/.venv/bin/python -m apollo_mavis_v2_hardware.netsetup install --yes \
    --arm view=192.168.2.219 --arm grip=192.168.1.201
# deploying for ANOTHER account (e.g. the operations user `mavis` whose venv lives
# under /opt): bake that venv's python into the hook and grant netdev to that user
# instead of $SUDO_USER; `--check` accepts the same two options.
sudo /path/to/.venv/bin/python -m apollo_mavis_v2_hardware.netsetup install --yes \
    --python /opt/apollo-mavis-v2/apollo-mavis-v2-hardware/.venv/bin/python --user mavis \
    --arm view=192.168.2.219 --arm grip=192.168.1.201
# hook + state dir only:
uv run python -m apollo_mavis_v2_hardware.netsetup install --dispatcher-only --arm ... --arm ...
```

## Notes

- State file precedence: `--state` > `~/.config/apollo-mavis-v2/nic_map.json`
  if present > `/etc/apollo-mavis-v2/nic_map.json` (root-written) if present.
- NM profiles (`/etc/NetworkManager/system-connections/*.nmconnection`) are
  shared by all accounts unless `connection.permissions` is set; `verify`
  warns and `reconcile` clears it.
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
