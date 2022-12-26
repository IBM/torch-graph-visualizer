#! /bin/bash

NCU=${1:-ncu}
OUTPUT=${2:-profile}

shift 2

set -x

"${NCU}" \
    --nvtx \
    --target-processes=all \
    --replay-mode=application \
    --call-stack \
    --force-overwrite \
    --export="${OUTPUT}" \
    "$@"

set +x
