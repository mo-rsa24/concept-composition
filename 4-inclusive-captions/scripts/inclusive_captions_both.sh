#!/usr/bin/env bash
# Run both arms of the pilot, one after the other, on one pinned device.
# A script file rather than an inline bash -c string: the inline form lost the newline between
# the two launcher lines and silently ran run A with run B's command as trailing arguments.
set -u
TAG="${1:?tag, e.g. v4}"
DEVICE="${2:-1}"
STEPS="${3:-2000}"
cd /home-mscluster/mmolefe/Playground/PhD/concept-composition-inclusive-captions || exit 1
bash 4-inclusive-captions/scripts/inclusive_captions_run.sh A "$DEVICE" "$STEPS" "$TAG"
echo "run A finished, starting run B" >&2
bash 4-inclusive-captions/scripts/inclusive_captions_run.sh B "$DEVICE" "$STEPS" "$TAG"
echo "both runs finished" >&2
