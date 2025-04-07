from pymatgen.core.structure import Structure
import json

with open("data/qmof_database/qmof_database/qmof_changed.json") as fp:
    targets = json.load(fp)

print(targets.keys())
for ke in targets.keys():
    print(targets[ke])
    break