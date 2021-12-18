import os
import json
import tqdm
from collections import OrderedDict



def create_vocab(path, name):
	utt = {}
	temp = None
	with open(path+name, "r") as f:
		queries = json.load(f)
	for i, dialog in tqdm.tqdm(queries.items()):
		for word in dialog["word_sequence_repunc_without_punctuation"].split():
			try:
				temp = utt[word]
			except KeyError:
				utt[word] = True	
	with open("kb_vocab.json", "w") as f:
		json.dump(utt, f, ensure_ascii=False, indent=4)

def create_oov(path, name, query=False):
	if os.path.exists("oov.json"):
		with open ("oov.json", "r") as f:
			temp_utt = json.load(f)
	else:
		temp_utt = dict()
	temp = None
	if query:
		key = "word_sequence"
	else:
 		key = "word_sequence_repunc_without_punctuation"
	with open("kb_vocab.json", "r") as f:
		utt = json.load(f)
	with open(path+name, "r") as f:
		queries = json.load(f)
	for i, dialog in tqdm.tqdm(queries.items()):
		for word in dialog[key].split():
			try:
				temp = utt[word]
			except KeyError:
				try:
					temp_utt[word] += 1
				except KeyError:
					temp_utt[word] = 1
	oov = [(key, temp_utt[key]) for key in sorted(temp_utt)]
	with open("oov_list.json", "w") as f:
		json.dump(oov, f, ensure_ascii=False, indent=4)
	with open("oov.json", "w") as f:
		json.dump(temp_utt, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
	kbs = [("/scratch/zw2374/public/es_exp/","DSTC9_task1_kb_eval_templates_es_sequences.json")]
	for kb in kbs:
		create_vocab(kb[0],kb[1])
	logs = [(
		"/scratch/zw2374/public/es_exp/",
		"DSTC9_task1_log_eval_templates_es_sequences.json"
		),
		("/scratch/zw2374/public/es_exp/",
		"DSTC9_task1_val_log_templates_es_sequences.json"
		),
		("/scratch/zw2374/public/es_exp/",
		"DSTC9_task1_train_logs_templates_es_sequences.json")]
	queries = [("/scratch/zw2374/public/es_exp/",
		"DSTC10_task2_log_templates_es_sequences.json"
		)]
	for log in logs:
		create_oov(log[0],log[1])
	for query in queries:
		create_oov(query[0],query[1],query=True)	
