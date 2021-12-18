import json
import os
import nltk
import re
import tqdm
from util_proc import normalize_punctuation, replace_numbers, replace_digits, prepare_disfluencies, remove_disfluencies


def remove_punctuation(query): #legacy
	temp = []
	found = query.find("''")
	while found != -1:
		query = query[:found] +	query[found+2:]
		found = query.find("''") 
	for c in query:
		if c in "(),.?!;\"\\":
			temp.append("")
		else:
			temp.append(c)
	results = "".join(temp).split()
	result = " ".join(["" if word in ["'", "-", "--", ":", "<", ">"] else word for word in results])
	return result

def create_clean(filename):
	with open(filename, "r") as f:
		raw = f.readlines()
	processed = []
	for entry in tqdm.tqdm(raw):
		new_entry = "".join(["." if ch == ";" else ch for ch in entry])
		found = new_entry.find(", and")
		while found != -1:
			new_entry = new_entry[:found]+ "."+ new_entry[found+1:]
			found = new_entry.find(", and")
		for sent in nltk.sent_tokenize(new_entry.strip()):
			sent = remove_punctuation(sent)
			sent = sent.strip()
			sent = replace_numbers(sent)
			sent = replace_digits(sent)
			processed.append(sent)
	with open("clean.json", "w") as f:
		json.dump(processed, f, ensure_ascii=False, indent=4)

def create_clean_enhanced(filename,target="clean_nocheat.json"):
	with open(filename, "r") as f:
		raw = f.readlines()
	processed = []
	for entry in tqdm.tqdm(raw):
		entry = normalize_punctuation(entry)
		entry = entry.strip()
		entry = replace_numbers(entry)
		entry = replace_digits(entry)

		processed.append(entry)
	with open(target, "w") as f:
		json.dump(processed, f, ensure_ascii=False, indent=4)

def create_clean_for_hannah(filename,target="clean_nocheat.json"):
	with open(filename, "r") as f:
		raw = f.readlines()
	processed = []
	for entry in tqdm.tqdm(raw):
		entry = entry[1:-1]
		entry = normalize_punctuation(entry)
		entry = entry.strip()
		processed.append(entry)
	with open(target, "w") as f:
		json.dump(processed, f, ensure_ascii=False, indent=4)

def create_ngram(filename, num_of_grams):
	window = num_of_grams
	utt_map = dict()
	results = []
	with open(filename, "r") as f:
		obj = json.load(f)
	for line in tqdm.tqdm(obj):
		word_list = line.split()
		for i in range(len(word_list)-1):
			temp = word_list[i:i+num_of_grams]
			max_char = -1
			min_char = 1000
			total_char = 0
			for word in temp:
				total += len(word)
				if len(word) > max_char:
					max_char = len(word)
				if len(word) < min_char:
					min_char = len(word)
			avg_char = total / num_of_grams
			temp += ["<\\s>"] * (num_of_grams - len(temp))
			ngram = " ".join(temp)
			if ngram in utt_map:
				utt_map[ngram]["freq"] += 1
			else:
				utt_map[ngram] = dict()
				utt_map[ngram]["freq"] = 1
				utt_map[ngram]["max_char"] = max_char
				utt_map[ngram]["min_char"] = min_char
				utt_map[ngram]["avg_char"] = avg_char
				utt_map[ngram]["len_map"] = [len(word) for word in temp]
				results.append(ngram)
	with open(str(num_of_grams)+ "_grams.json", "w") as f:
		json.dump(sorted(results), f, ensure_ascii=False, indent=4)
	with open(str(num_of_grams)+ "_grams_map.json", "w") as f:
		json.dump(utt_map, f, ensure_ascii=False, indent=4)

def create_ngram_enhanced(filename, num_of_grams):
	window = num_of_grams
	utt_map = dict()
	results = []
	with open(filename, "r") as f:
		obj = json.load(f)
	for line in tqdm.tqdm(obj):
		word_list = line.split()
		for i in range(len(word_list)-1):
			temp = word_list[i:i+num_of_grams]
			if len(temp) < num_of_grams:
				break
			max_char = -1
			min_char = 1000
			total_char = 0
			for word in temp:
				total_char += len(word)
				if len(word) > max_char:
					max_char = len(word)
				if len(word) < min_char:
					min_char = len(word)
			avg_char = total_char / num_of_grams
			ngram = " ".join(temp)
			if ngram in utt_map:
				utt_map[ngram]["freq"] += 1
			else:
				utt_map[ngram] = dict()
				utt_map[ngram]["freq"] = 1
				utt_map[ngram]["max_char"] = max_char
				utt_map[ngram]["min_char"] = min_char
				utt_map[ngram]["avg_char"] = avg_char
				utt_map[ngram]["len_map"] = [len(word) for word in temp]
				results.append(ngram)
	with open(str(num_of_grams)+ "_grams.json", "w") as f:
		json.dump(sorted(results), f, ensure_ascii=False, indent=4)
	with open(str(num_of_grams)+ "_grams_map.json", "w") as f:
		json.dump(utt_map, f, ensure_ascii=False, indent=4)
	
					

if __name__ == "__main__":
	print("processing clean text...")
	#create_clean_for_hannah("raw_sentence.txt","raw_sentence_processed.json")
	create_clean_enhanced("final.txt")
	for i in range(1,7):
		print("creating",i,"gram(s)...")
		create_ngram_enhanced("clean_nocheat.json",i)
