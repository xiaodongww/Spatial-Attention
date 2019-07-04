#!/usr/bin/env bash
#python main.py -d market -b 48 -j 4 --epochs 100 --log logs/market/ --combine-trainval --step-size 40 --data-dir ./data/market1501
#python main.py -d market -b 48 -j 4 --epochs 150 --log logs/market_resume/ --combine-trainval --step-size 40 --data-dir ./data/market1501 --resume logs/market/checkpoint.pth.tar
python main.py -d market -b 48 -j 4 --epochs 150 --log logs/market_epoch150/ --combine-trainval --step-size 40 --data-dir ./data/market1501