#!/usr/bin/env bash
python main.py -d market -b 48 -j 4 --epochs 100 --log logs/market/ --combine-trainval --step-size 40 --data-dir ./data/market1501