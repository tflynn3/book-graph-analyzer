"""Check timeline data."""
import json

with open('data/output/hobbit_ch1_timeline.json') as f:
    t = json.load(f)

print(f"Entities: {len(t.get('entities', {}))}")
print(f"Relations: {len(t.get('relations', []))}")
print()

print("Entities with era info:")
for name, e in list(t.get('entities', {}).items())[:10]:
    birth = e.get('birth_era', '?')
    death = e.get('death_era', '?')
    print(f"  {name}: {birth} - {death}")

print()
print("Sample temporal relations:")
for r in t.get('relations', [])[:15]:
    print(f"  {r['subject']} --{r['relation']}--> {r['object']}")
