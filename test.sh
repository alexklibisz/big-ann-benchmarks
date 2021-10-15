#!/bin/bash
set -e

DATASET=deep-10M
#DATASET=msturing-1M

sudo chown -R $(whoami):$(whoami) results/

cd ../elastiknn && ./gradlew shadowJar && cd -
mv ../elastiknn/elastiknn-ann-benchmarks/build/libs/ann-benchmarks-7.14.1.1-all.jar .

export LIBRARY=elastiknn
venv/bin/python install.py

python create_dataset.py --dataset $DATASET

python run.py --algorithm elastiknn-t2 --dataset $DATASET --runs 1 --count 100
sudo chown -R $(whoami):$(whoami) results/
python plot.py --dataset $DATASET --recompute --count 100 -o out.png
