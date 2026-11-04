# AGENTS.md

Notes for agents working in this repository.

## Harness environment: the SSH config-permission error is a HARNESS issue

Running `ssh` (and therefore `git push` over SSH) from this harness can fail
with:

```
Bad owner or permissions on /etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf
fatal: Could not read from remote repository.
```

**This is a harness issue — there is NO problem with the config protections.**
The ownership/permission rules that ssh applies to files under
`/etc/ssh/ssh_config.d/` are correct as designed. Do **not** try to "fix" the
symlink's ownership or permissions, do not modify `/etc/ssh/ssh_config` or its
`Include` line, and do not treat the system ssh configuration as broken. The
harness environment is what makes ssh reject that include.

Work around it for the push itself, leaving the system config alone:

```sh
GIT_SSH_COMMAND='ssh -F /dev/null' git push origin main
```

`-F /dev/null` skips the system-wide config (and therefore the problematic
`Include`) while still using the default identity files (`~/.ssh/id_rsa`) and
`~/.ssh/known_hosts`; authentication as the repository owner works normally.

### Related harness sandbox limits when pushing

The pre-push hook (`.githooks/pre-push`) works as written on a normal
machine; from inside the harness these env overrides are needed because
`~/.cache/uv` and `/var/tmp` are outside the writable area:

```sh
UV_CACHE_DIR="$PWD/.uv-cache" \
PYTEST_ADDOPTS="--basetemp=$PWD/.pytest-tmp" \
GIT_SSH_COMMAND='ssh -F /dev/null' \
git push origin main
```

(`PYTEST_ADDOPTS` works because the hook's own `--basetemp` is prepended and
the last occurrence wins; both scratch directories are git-excluded.)
