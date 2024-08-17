#!/bin/bash


for dir in ./youtube8m/*/*; do
    echo $dir
    for file in $dir/*.wav; do
        mv -n $file ./y8m_audios/
    done
done

echo "Total complete"

