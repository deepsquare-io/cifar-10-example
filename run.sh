#!/bin/sh

podman run \
  --rm \
  --device nvidia.com/gpu=all \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/checkpoint:/checkpoint" \
  -u 1000:1000 \
  --group-add keep-groups \
  --entrypoint /bin/sh \
  ghcr.io/deepsquare-io/cifar-10-example:latest \
  -c '\
  mpirun \
  -np 4 \
  /.venv/bin/python3 \
  /app/main.py \
  --no-cuda \
  --horovod \
  --checkpoint_in=/checkpoint/ckpt.pth \
  --checkpoint_out=/checkpoint/ckpt.pth \
  --dataset=/data
'
