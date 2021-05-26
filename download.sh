#!/bin/bash

echo "Downloading attribute classifier checkpoint"
wget -O models/attribute_classifier/BranchedTiny.ckpt https://www.dropbox.com/s/nbm6oi70x158sn3/BranchedTiny.ckpt?dl=1  # mogoce dl=1?

echo "Downloading face parser checkpoint"
wget -O models/face_parser/FaceParser.ckpt https://www.dropbox.com/s/oe45ovn0we7ynh8/FaceParser.ckpt?dl=1

echo "Done"