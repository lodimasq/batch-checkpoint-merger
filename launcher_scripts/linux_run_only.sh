#!/usr/bin/env bash
source "./venv/bin/activate"
nohup python -m batch_checkpoint_merger >/dev/null 2>&1 &