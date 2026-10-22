#!/bin/bash
# CI entry point. All case selection and parametrization lives in the pytest
# files; this only maps a group name to a pytest marker expression.
#   test/unit_test.sh [fwd_bwd | load_dump | unit | integration]   (from corelib/dynamicemb)
# Extra arguments go to pytest, e.g. `test/unit_test.sh unit -k lru`.
# See test/README.md for the layout and the catalog of what each test covers.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

case "${1:-all}" in
    fwd_bwd|load_dump|unit|integration) MARK=(-m "$1"); shift ;;
    *) MARK=() ;;
esac

exec python -m pytest "${MARK[@]}" "$@"
