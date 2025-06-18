#!/bin/bash

# Upgrade pip and install base requirements
python -m pip install --upgrade pip
python -m pip install --upgrade wheel setuptools packaging

# Install the requirements with specific options for newer Python versions
export SETUPTOOLS_USE_DISTUTILS=stdlib
python -m pip install --upgrade -r requirements.txt --no-cache-dir

echo "Setup completed successfully!"
