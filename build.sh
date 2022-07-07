#!/bin/bash

docker build . --platform linux/amd64 --tag featurecloud.ai/fc_cross_validation
# docker build . --platform linux/amd64 --tag registry.featurecloud.eu:5000/fc_cross_validation
