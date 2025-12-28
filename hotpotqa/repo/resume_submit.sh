#!/bin/bash
set -euo pipefail

echo "Submitting resume-friendly HotpotQA compare job..."
jid=$(sbatch --parsable job.hotpot_compare_resume.sbatch)
echo "Submitted: $jid"

echo
echo "Follow logs:"
echo "  tail -f hotpot_gepa_cmp.${jid}.out"
echo
echo "Live outputs folder:"
echo "  $WORK/gepa_hotpotqa/runs/latest/logs"
echo
echo "Ports used (once job starts):"
echo "  cat $WORK/gepa_hotpotqa/runs/latest/ports.txt"
