#!/bin/bash
# filepath: /raid/home/dgx1009/facies-classification/setup_data.sh

# Download the facies classification dataset
echo "Downloading data.zip from Zenodo..."
wget https://zenodo.org/record/3755060/files/data.zip

# Verify the MD5 checksum
echo "Verifying MD5 checksum..."
checksum=$(openssl dgst -md5 data.zip)
expected="MD5(data.zip)= bc5932279831a95c0b244fd765376d85"

if [ "$checksum" = "$expected" ]; then
    echo "✓ Checksum verified successfully"
else
    echo "✗ Checksum mismatch! Expected: $expected"
    echo "✗ Got: $checksum"
    echo "The downloaded data.zip may be corrupted. Please try downloading again."
    exit 1
fi

# Extract the data
echo "Extracting data.zip..."
unzip data.zip

# Create splits directory
echo "Creating splits directory..."
mkdir -p data/splits

echo "Setup complete!"