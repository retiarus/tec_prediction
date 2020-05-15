#!/bin/bash

fileid="17POAs_3-Nssyb1sZdEIcRlf5_g4O7x3F"
filename="re-tecmap.tar.xz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
