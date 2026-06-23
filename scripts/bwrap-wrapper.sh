#!/bin/bash
# bwrap wrapper for nix-portable on systems with domain accounts (AD/Winbind/LDAP/SSSD).
#
# On managed systems the current user may not appear in the local /etc/passwd —
# they live in a directory service instead. Inside nix-portable's bwrap container
# the directory service socket is unreachable, so getpwuid() returns nothing and
# SSH cannot find the user's home directory or key.
#
# This wrapper injects the current user's passwd entry (resolved before the
# container starts, while the directory service is still reachable) into a
# temporary passwd file that is bind-mounted over /etc/passwd inside bwrap.
#
# Safe to use on normal machines too: if the user is already in /etc/passwd the
# grep finds them and nothing is appended — effectively a no-op.
#
# Usage: export NP_BWRAP=/path/to/bwrap-wrapper.sh

TMPPASSWD=$(mktemp /tmp/nix-passwd.XXXXXX)
cp /etc/passwd "$TMPPASSWD"
CURRENT_UID=$(id -u)
if ! grep -q "^[^:]*:[^:]*:${CURRENT_UID}:" "$TMPPASSWD" 2>/dev/null; then
    ENTRY=$(getent passwd "$CURRENT_UID" 2>/dev/null)
    [ -n "$ENTRY" ] && echo "$ENTRY" >> "$TMPPASSWD"
fi
trap "rm -f '$TMPPASSWD'" EXIT

NEWARGS=()
i=0
ARGS=("$@")
while [ $i -lt ${#ARGS[@]} ]; do
    arg="${ARGS[$i]}"
    if [ "$arg" = "--bind" ] && [ $((i+2)) -lt ${#ARGS[@]} ] \
       && [ "${ARGS[$((i+1))]}" = "/etc/passwd" ]; then
        NEWARGS+=("--bind" "$TMPPASSWD" "${ARGS[$((i+2))]}")
        i=$((i+3))
    else
        NEWARGS+=("$arg")
        i=$((i+1))
    fi
done

exec /usr/bin/bwrap "${NEWARGS[@]}"
