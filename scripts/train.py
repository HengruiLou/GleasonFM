#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from gleason_proto.utils.config import load_config
from gleason_proto.engine.train_loop import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to train yaml config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg)

if __name__ == "__main__":
    main()
