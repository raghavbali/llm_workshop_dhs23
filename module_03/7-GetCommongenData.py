import json, random

random.seed(999)

data_path = "data/commongen_data/"
filenames = [data_path + "commongen.train.jsonl", data_path + "commongen.dev.jsonl"]
out_path = "data/"

data = []
for filename in filenames:
    with open(filename) as f:
        for line in f.readlines():
            data.append(json.loads(line))
train_set = []
val_set = []
for item in data:
    concepts = item["concept_set"]
    scenes = item["scene"]
    if random.random() < 0.8:
        for scene in scenes:
            train_set.append([concepts, scene])
    else:
        for scene in scenes:
            val_set.append([concepts, scene])
random.shuffle(train_set)
random.shuffle(val_set)
with open(out_path + "commongen_train.bin", "w", encoding="utf8") as f:
    f.write("\n".join(["@".join(entry) for entry in train_set]) + "\n")
with open(out_path + "commongen_val.bin", "w", encoding="utf8") as f:
    f.write("\n".join(["@".join(entry) for entry in val_set]) + "\n")
