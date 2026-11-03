"""Summarize opt-in Shiny traces: python -m apmtools.shiny.profile_report trace.jsonl"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(path):
    groups = defaultdict(list)
    failures = defaultdict(int)
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get('event') != 'execution':
            continue
        key = (record.get('kind', 'unknown'), record.get('name', 'unknown'))
        groups[key].append(float(record.get('duration_ms', 0)))
        if record.get('error'):
            failures[key] += 1
    rows = []
    for key, durations in groups.items():
        ordered = sorted(durations)
        rows.append({'kind': key[0], 'name': key[1], 'calls': len(ordered),
                     'total_ms': round(sum(ordered), 2),
                     'mean_ms': round(sum(ordered)/len(ordered), 2),
                     'p95_ms': ordered[min(len(ordered)-1, int(.95*(len(ordered)-1)))],
                     'max_ms': ordered[-1], 'errors': failures[key]})
    return sorted(rows, key=lambda r: r['total_ms'], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', help='Path to APMTOOLS_TRACE_PATH JSONL file')
    parser.add_argument('--top', type=int, default=35)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    rows = summarize(args.trace)
    if args.json:
        print(json.dumps(rows[:args.top], indent=2))
        return
    print(f"{'kind':17} {'name':37} {'calls':>6} {'total ms':>10} {'p95 ms':>10} {'errors':>6}")
    for r in rows[:args.top]:
        print(f"{r['kind'][:17]:17} {r['name'][:37]:37} {r['calls']:6} {r['total_ms']:10.2f} {r['p95_ms']:10.2f} {r['errors']:6}")

if __name__ == '__main__':
    main()
