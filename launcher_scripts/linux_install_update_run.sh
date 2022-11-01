#!/usr/bin/env bash
source "./venv/bin/activate"
python -m pip install --upgrade batch_checkpoint_merger
nohup python -m batch_checkpoint_merger >/dev/null 2>&1 &
