#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
exec "${ROBOT_PYTHON:-python3}" -u app.py "$@"
