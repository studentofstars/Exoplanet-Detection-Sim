#!/bin/bash

# Upgrade pip and install base requirements
python -m pip install --upgrade pip wheel setuptools

# Install the requirements
python -m pip install -r requirements.txt --no-cache-dir

echo "Setup completed successfully!"
