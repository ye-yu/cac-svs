#!/bin/bash

for i in *.wav;
    do name=`echo "$i" | cut -d'.' -f1`
    echo "Amplifying $i into /wav/{$name}.wav"
    ffmpeg -i "$i" -af "dynaudnorm=p=0.95:m=100:g=15" -y "./amplified/${name}.wav"
done
