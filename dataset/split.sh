#! /usr/bin/env bash

cat parsed/*.jsonl > methods.jsonl

sed -i '/^[[:space:]]*$/d' methods.jsonl

shuf methods.jsonl > methods_shuffled.jsonl
split -n l/10 methods_shuffled.jsonl

cat xa{a,b,c,d,e,f,g,h} > train.jsonl
mv xai val.jsonl
mv xaj test.jsonl

rm xa*

