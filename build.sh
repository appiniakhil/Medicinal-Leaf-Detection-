#!/bin/bash

# Update MariaDB Connector/C to the required version
# Note: Replace 'apt-get' with the appropriate package manager for your OS.

# Check if the required version is already installed
if ! dpkg -l | grep -q "libmariadb3" && ! dpkg -l | grep -q "libmariadb-dev"; then
  echo "Updating MariaDB Connector/C..."
  sudo apt-get update
  sudo apt-get install -y libmariadb3 libmariadb-dev
else
  echo "Required version of MariaDB Connector/C is already installed."
fi
set -o errexit
sudo apt-get install libmariadb3 libmariadb-dev
pip install --upgrade pip
pip install -r requirements.txt