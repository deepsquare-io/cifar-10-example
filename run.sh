#!/bin/sh

podman run \
  --rm \
  --device nvidia.com/gpu=all \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/checkpoint:/checkpoint" \
  cifar-10-example:latest \
  --checkpoint_in=/checkpoint/ckpt.pth \
  --checkpoint_out=/checkpoint/ckpt.pth \
  --dataset=/data
