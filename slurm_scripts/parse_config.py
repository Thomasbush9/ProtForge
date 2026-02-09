#!/usr/bin/env python3
"""Parse YAML config file and output variables for bash script"""
import sys
import yaml
import json

if len(sys.argv) < 3:
    print("Usage: parse_config.py CONFIG_FILE KEY", file=sys.stderr)
    sys.exit(1)

config_file = sys.argv[1]
key = sys.argv[2]

try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Navigate nested keys (e.g., "input.fasta_dir")
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            print("", file=sys.stdout)
            sys.exit(1)
    
    # Output value (handle None as empty string)
    if value is None:
        print("", file=sys.stdout)
    else:
        print(str(value), file=sys.stdout)
except Exception as e:
    print(f"Error parsing config: {e}", file=sys.stderr)
    sys.exit(1)














