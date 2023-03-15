#!/bin/bash

echo "Downloading attribute classifier checkpoint"
wget -O models/attribute_classifier/state.pt https://www.dropbox.com/s/4j1x9zhri8cndny/state.pt?dl=1

echo "Downloading face parser checkpoint"
wget -O models/face_parser/state.pt https://www.dropbox.com/s/2mwswfq42xvhied/state.pt?dl=1

echo "Downloading e4e encoder"
wget -O models/e4e/state.pt https://www.dropbox.com/s/t8tm5tbk27wa7er/state.pt?dl=1

echo "Done"
