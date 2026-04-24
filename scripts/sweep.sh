#!/bin/bash
# DOC2TXT — OCD project-specific sweep checks.
#
# Sourced by ~/.claude/scripts/ocd-sweep.sh (the generic runner).
# See ~/.claude/fgt/templates/SWEEP_TEMPLATE.sh for the format spec.
#
# Per Principle 1 (Every Addition Must Justify Its Keep), add project-
# specific SWEEP_PROCESSES / SWEEP_DATA_CHECKS entries only when a bug
# class has been observed >=2 times and a sweep check is the right
# enforcement plane for it. Empty arrays here are intentional; the
# generic runner handles git, BACKLOG, .env perms automatically.

SWEEP_PROCESSES=()
SWEEP_DATA_CHECKS=()
SWEEP_DB=""
