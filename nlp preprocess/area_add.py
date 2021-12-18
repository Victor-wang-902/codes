import json


with open("/scratch/jx1038/public/dstc10_test_idmap.json", "r") as f:
	obj1 = json.load(f)

with open("system_final.json", "r") as f:
	obj2 = json.load(f)
for i, x in enumerate(obj1):
	for j, y in enumerate(x):
		obj1[i][j]["area_map"] = obj2[i][j]["area_map"] 
with open("system_area_final.json", "w") as f:
	json.dump(obj1, f, ensure_ascii=False,indent=4)

