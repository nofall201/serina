#!/bin/bash
# Install gdown (pastikan pip ada)
pip install --upgrade gdown

mkdir -p bin

# Download dari Google Drive (pakai file ID dari link kamu)
gdown --id 1g59xCIiSE6WT-rLFILzs0KVIIf2H7Mx6 -O bin/ffmpeg
gdown --id 1zfQE4FaEzPp0Z5J9Ajn-aALSo5-ZacWH -O bin/ffprobe

chmod +x bin/ffmpeg bin/ffprobe
