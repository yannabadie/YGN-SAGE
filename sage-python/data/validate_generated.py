
import json, os, sys
from pathlib import Path

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
files = {
    'error_correction': base / 'gpt54_error_correction.jsonl',
    'preferences': base / 'gpt54_preferences.jsonl',
    'deep_reasoning': base / 'gpt54_deep_reasoning.jsonl',
    'simple_calibrated': base / 'gpt54_simple_calibrated.jsonl',
    'audit': base / 'gpt54_audit.jsonl',
}

def check_topology(topo):
    assert isinstance(topo, dict), 'topology is not an object'
    nodes = topo.get('nodes', [])
    assert isinstance(nodes, list) and len(nodes) >= 1, 'no nodes'
    assert nodes[-1].get('role') == 'synthesizer', f'last node is {nodes[-1].get("role")}, not synthesizer'
    return True

for name, path in files.items():
    if not path.exists():
        print(f'{name}: NOT FOUND')
        continue
    total = 0
    valid = 0
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            total += 1
            e = json.loads(line)
            try:
                if name == 'error_correction':
                    check_topology(e['topology_v2'])
                elif name == 'preferences':
                    check_topology(e['topology_a'])
                    check_topology(e['topology_b'])
                elif name == 'audit':
                    check_topology(e['improved'])
                else:
                    check_topology(e['topology'])
                valid += 1
            except Exception as ex:
                print(f'  {name} line {i}: {ex}')
    print(f'{name}: {valid}/{total} valid')
