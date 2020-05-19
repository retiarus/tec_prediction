#!/bin/bash

fileid="1HclMxP_w5zh50DKuLBhgwPvBMuiwmHjz"
filename="resized_type_2.tar.xz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
