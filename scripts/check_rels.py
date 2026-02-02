import json
with open('data/exports/hobbit_relationships.json') as f:
    d = json.load(f)
print('Total records:', len(d.get('relationships', [])))
rels_with_ids = 0
for rec in d.get('relationships', []):
    for r in rec.get('relationships', []):
        if r.get('subject_id') and r.get('object_id'):
            rels_with_ids += 1
            if rels_with_ids <= 10:
                print(f"  {r['subject_id']} -> {r['object_id']}: {r['predicate']}")
print(f'Total with both IDs: {rels_with_ids}')
