#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from gleason_proto.utils.config import load_config
from gleason_proto.engine.eval_loop import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to eval yaml config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)

if __name__ == "__main__":
    main()
