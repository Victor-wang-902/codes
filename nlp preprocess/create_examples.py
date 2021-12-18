from punctuator import Punctuator
import json
import os
import tqdm


def raw_punc(method_name, target_name):
	new_entries = []
	with open("sample_text.json", "r") as f:
		entries = json.load(f)
	method = Punctuator(method_name)
	for item in tqdm.tqdm(entries):
		new_entries.append(method.punctuate(item))
	with open(target_name, "w") as f:
		json.dump(new_entries, f, ensure_ascii=False, indent=4)

def create_sample_texts():
	new_entries = []
	with open("DSTC10_task2_log_templates_es_sequences.json", "r") as f:
		entries = json.load(f)
	for num, entry in entries.items():
		new_entry = entry["word_sequence"]
		new_entries.append(new_entry)
		if num == "100":
			break
	with open("sample_text.json", "w") as f:
		json.dump(new_entries,f, ensure_ascii=False, indent=4)

def punctuate_text(method_name, target_name):
	new_entries = []
	with open("sample_text.json", "r") as f:
		entries = json.load(f)
	count=0
	method = Punctuator(method_name)
	for item in entries:
		print(count)
		count+=1
		new_entries.append(punctuator_wrapper(item, method))
	with open(target_name, "w") as f:
		json.dump(new_entries, f, ensure_ascii=False, indent=4)
		
def filter_markers(markers, threshold):
	n = 0
	while n < len(markers):
		if n == 0:
			diff = markers[n] + 1
		else:
			diff = markers[n] - markers[n-1]
		if diff <= threshold:
			if n == len(markers) - 1:
				markers.pop(n-1)
			else:
				markers.pop(n)
		else:
			n += 1
	return markers

def punctuator_wrapper(entry, method=Punctuator("Demo-Europarl-EN.pcl")):
	new = punctuate(method,entry)
	markers = make_markers(new)
	return apply_markers(markers, new)

def make_markers(entry):
	markers = []
	for n, word in enumerate(entry.split()):
		for ch in word:
			if ch in ";?!,.":
				markers.append(n)
				break
	markers = filter_markers(markers, 3)
	return markers
	

def punctuate(method, item):
	new_entry = method.punctuate(item)
	new = ""
	for ch in new_entry:
		if ch not in "-:":
			new += ch
	return new

def apply_markers(markers, entry):
	modified = []
	for n, word in enumerate(entry.split()):
		temp = ""
		if n not in markers:
			for ch in word:
				if ch not in ",.?!;":
					temp += ch
		else:
			for ch in word:
				if ch in ",;":
					temp += "."
				else:
					temp += ch
		modified.append(temp)
	return " ".join(modified)

if __name__ == "__main__":
	create_sample_texts()
	punctuate_text("Demo-Europarl-EN.pcl", "euro.json")
	punctuate_text("INTERSPEECH-T-BRNN.pcl", "inter.json")
	raw_punc("Demo-Europarl-EN.pcl", "euro_raw.json")
