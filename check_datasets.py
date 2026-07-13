from datasets import load_dataset

arc = load_dataset('ai2_arc', 'ARC-Easy', split='test')
print('ARC columns:', arc.column_names)
ex = arc[0]
for k,v in ex.items():
    print(f'  {k}: {repr(v)[:120]}')
print('ARC size:', len(arc))

print()
sqa = load_dataset('basicv8vc/SimpleQA', split='test')
print('SimpleQA cols:', sqa.column_names)
ex2 = sqa[0]
for k,v in ex2.items():
    print(f'  {k}: {repr(v)[:100]}')
print('SimpleQA size:', len(sqa))
