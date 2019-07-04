#!/usr/bin/env bash
#python main.py -d market -b 48 -j 4 --log logs/market_authors/ --combine-trainval --step-size 40 --data-dir ./data/market1501 --resume logs/market_authors/checkpoint.pth.tar --evaluate
python main.py -d market -b 48 -j 4 --log logs/market_reproduce/ --combine-trainval --step-size 40 --data-dir ./data/market1501 --resume logs/market_epoch150/model_best.pth.tar --evaluate