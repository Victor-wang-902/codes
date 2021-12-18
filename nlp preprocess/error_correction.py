from elastic_search import search_retrieve_wrapper
from util_tree import TreeNode
from util_proc import replace_numbers, replace_digits
import json
from tqdm import tqdm


def correction_wrapper(es, to_correct_path, result_path="correction_results.json", method="fixed", size=50, show_result=10):
	result = dict()
	with open(to_correct_path, "r") as f:
		puzzles = json.load(f)
	for i, turn in tqdm(enumerate(puzzles)):
		for j, line in enumerate(turn):
			if line["speaker"] == "S":
				continue
			sentence = line["new_text"]	
			sentence = replace_numbers(sentence)
			result[sentence] = dict()
			result[sentence]["corrections"] = []
			error_regions = line["new_error_region"]
			result[sentence]["error_region"] = error_regions
			res = fixed_length_algorithm(es, sentence, error_regions, size=size)
			for item in res:
				result[sentence]["corrections"].append(item[:show_result])
	with open(result_path, "w") as f:
		json.dump(result, f, ensure_ascii=False, indent=4)

def error_region_scan(puzzle, error_region):
	error = False
	puzzle_list = puzzle.split()
	queries = []
	temp = []
	for i, token in enumerate(error_region):
		if not error:
			if token == 1:
				error = True
				if i == 0:
					temp.append("<\\s>")
				else:
					temp.append(puzzle_list[i-1])
				temp.append(puzzle_list[i])
		else:
			temp.append(puzzle_list[i])
			if token == 0:
				error = False
				queries.append(" ".join(temp))
				temp = []
			else:
				continue
	if len(temp) != 0:
		temp.append("<\\s>")
		queries.append(" ".join(temp))
	return queries		


def dynamic_algorithm(es, puzzle, error_region, num_of_gram=2, size=1000, wild_start_node_limit=100, dynamic_branch_size=6, sweep_comb="union"):
	assert num_of_gram in [2,3,4,5]
	max_limit = 5 * size
	max_dynamic_limit = 5 * dynamic_branch_size
	def _ngramize(words):
		res = []
		for i in range(len(words) - num_of_grams + 1):
			ngram = word[i:i + num_of_grams]
			if "<\\s>" in ngram.split():
				continue
			res.append(ngram)
		return res
	def _get_signal(words):
		start_signal = False if words[0] == "<\\s>" else True
		end_signal = False if words[-1] == "<\\s>" else True
		return start_signal, end_signal
	def _dynamic_search(es, query, var = 1):
		assert var < num_of_gram and var + num_of_gram <= 6
		lower =  search_retrieve_wrapper(es, query, num_of_gram - var, size=max_dynamic_limit)
		upper =  search_retrieve_wrapper(es, query, num_of_gram + var, size=max_dynamic_limit)
		search_res = search_retrieve_wrapper(es, query, num_of_gram, size=max_limit) + lower + upper
		search_res.sort(key=lambda x: x[0], reverse=True)
		return search_res
	def _fixed_search(es, ngrams):
		meta = []
		for ngram in ngrams:
			search_res = search_retrieve_wrapper(es, " ".join(ngram), num_of_grams, size=max_limit)
			search_res.sort(key=lambda x: x[0], reverse=True)
			meta.append(search_res)
		return meta
	def _long_sweep(ngrams, start_signal, end_signal, direction=0):
		pass
	def _dynamic_sweep(ngrams, start_signal, end_signal, direction=0):
		result = []
		new_leaves = []
		root = TreeNode((0, ""))
		if not direction:
			if not start_signal:
				leaves = [root]
			else:
				temp = TreeNode((0, words[0]))
				root.add_child(temp)
				leaves = [temp]
			for i, ngram in enumerate(ngrams):
				end = True if i == len(ngrams) - 1 else False
				for leaf in leaves:
					if leaf.data[1] != "":
						first = leaf.data[1]
						query = first + " " + " ".join(ngram.split()[1:])
						dynamic_candidates = _dynamic_search(es, query)
						for res in dynamic_candidates:
							candidate = res[1].split()
							if not end or not signal:
								if candidate[0] == leaf.data[1].split()[-1]:
									new_leaf = TreeNode((res[0], res[1]))
									leaf.add_child(new_leaf)
									new_leaves.append(new_leaf)
							else:
								if candidate[0] == leaf.data[1].split()[-1] and candidate[-1] == words[-1]:
									new_leaf = TreeNode((res[0], res[1]))
									leaf.add_child(new_leaf)
									new_leaves.append(new_leaf)
				leaves = new_leaves[:]
				new_leaves = []
		else:
			if not end_signal:
				leaves = [root]
			else:
				temp = TreeNide((0,words[0]))
				root.add_child()
			if words[-1] == "<\\s>":
				leaves = []
				search_res = search_retrieve_wrapper(es, words[-1], 1, size=wild_start_node_limit)
				for res in search_res:
					temp = TreeNode((res[0], res[1]))
					root.add_child(temp)
					leaves.append(temp)
			else:
				temp = TreeNode((0, words[-1]))
				root.add_child(temp)
				leaves = [temp]
			for i, search_res in enumerate(meta_res[::-1]):
				end = True if i == len(meta_res) - 1 else False
				for leaf in leaves:
					for res in search_res:
						candidate = res[1].split()
						if signal and end:
							if candidate[-1] == leaf.data[1].split()[0] and candidate[0] == words[0]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
						else:
							if candidate[-1] == leaf.data[1].split()[0]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
				leaves = new_leaves[:]
				new_leaves = []
		for leaf in leaves:
			current = leaf
			str_buffer = []
			score_buffer = []
			while current is not None:
				score_buffer.append(current.data[0])
				if current.parent is not root:
					if not direction:
						str_buffer.append(" ".join(current.data[1].split()[1:]))
					else:
						str_buffer.append(" ".join(current.data[1].split()[:-1]))
				else:
					str_buffer.append(current.data[1])
					break
				current = current.parent
			if not direction:
				result_str = " ".join(str_buffer[::-1])
			else:
				result_str = " ".join(str_buffer)
			total_score = 0
			for score in score_buffer:
				total_score += score
			result.append((result_str, total_score))
		result.sort(key=lambda x: x[0], reverse=True)
		return result[:max_limit]

	queries = error_region_scan(puzzle, error_region)
	corrections = []
	for query in queries:
		words = query.split()
		if len(words) > 7:
			continue
		ngrams = _ngramize(words)
		start_sig, end_sig = _get_signal(words)
		meta_res = _fixed_search(es, words)
		left_sweep_res = _sweep(words, end_signal, meta_res)
		right_sweep_res = _sweep(words, start_signal, meta_res, 1)
		if sweep_comb == "union":
			res = []
			left_dict = {text:score for text, score in left_sweep_res}
			for item in right_sweep_res:
				if item[0] not in left_dict:
					res.append((item[0], item[1]))
				else:
					res.append((item[0], (item[1] + left_dict[item[0]]) / 2))
					left_dict.pop(item[0])
			for key, value in left_dict.items():
				res.append((key, value))
			res.sort(key=lambda x: x[1], reverse=True)
		elif sweep_comb == "intersection":
			res = []
			left_dict = {text:score for text, score in left_sweep_res}
			for item in right_sweep_res:
				if item[1] in left_dict:
					res.append((item[0] + left_dict[item[1]] / 2, item[1]))
		corrections.append(res[:size])
	return corrections

			
def fixed_length_algorithm(es, puzzle, error_region, num_of_grams=2, size=1000, wild_start_node_limit=100, sweep_comb="union"):
	max_limit = 10 * size
	if wild_start_node_limit > max_limit:
		wild_start_node_limit = max_limit
	def _search(es, words):
		start_signal = False if words[0] == "<\\s>" else True
		end_signal = False if words[-1] == "<\\s>" else True
		meta = []
		for i in range(len(words) - num_of_grams + 1):
				ngram = words[i:i + num_of_grams]
				if not i:
					if not start_signal:
						continue
				elif i == len(words) - num_of_grams:
					if not end_signal:
						break
				search_res = search_retrieve_wrapper(es, " ".join(ngram), num_of_grams, size=max_limit)
				search_res.sort(key=lambda x: x[0], reverse=True)
				meta.append(search_res)
		if not meta:
			search_res = search_retrieve_wrapper(es, words[1], 1,size=max_limit) 
			search_res.sort(key=lambda x: x[0], reverse=True)
			meta.append(search_res)

		return meta, start_signal, end_signal

	def _sweep(words, signal, meta_res, direction=0):
		result = []
		new_leaves = []
		root = TreeNode((0, ""))
		if not direction:
			if words[0] =="<\\s>":
				leaves = [] 
				search_res = meta_res[0][:wild_start_node_limit]	
				for res in search_res:
					temp = TreeNode((res[0], res[1].split()[0]))
					root.add_child(temp)
					leaves.append(temp)	
			else:
				temp = TreeNode((0, words[0]))
				root.add_child(temp)
				leaves = [temp]
			for i, search_res in enumerate(meta_res):
				end = True if i == len(meta_res) - 1 else False
				for leaf in leaves:
					for res in search_res:
						candidate = res[1].split()
						if not end or not signal:
							if candidate[0] == leaf.data[1].split()[-1]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
						else:
							if candidate[0] == leaf.data[1].split()[-1] and candidate[-1] == words[-1]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
				leaves = new_leaves[:]
				new_leaves = []
		else:
			if words[-1] == "<\\s>":
				leaves = []
				search_res = meta_res[-1][:wild_start_node_limit]	
				for res in search_res:
					temp = TreeNode((res[0], res[1].split()[-1]))
					root.add_child(temp)
					leaves.append(temp)
			else:
				temp = TreeNode((0, words[-1]))
				root.add_child(temp)
				leaves = [temp]
			for i, search_res in enumerate(meta_res[::-1]):
				end = True if i == len(meta_res) - 1 else False
				for leaf in leaves:
					for res in search_res:
						candidate = res[1].split()
						if signal and end:
							if candidate[-1] == leaf.data[1].split()[0] and candidate[0] == words[0]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
						else:
							if candidate[-1] == leaf.data[1].split()[0]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
				leaves = new_leaves[:]
				new_leaves = []
		for leaf in leaves:
			current = leaf
			str_buffer = []
			score_buffer = []
			while current is not None:
				score_buffer.append(current.data[0])
				if current.parent is not root:
					if not direction:
						str_buffer.append(" ".join(current.data[1].split()[1:]))
					else:
						str_buffer.append(" ".join(current.data[1].split()[:-1]))
				else:
					str_buffer.append(current.data[1])
					break
				current = current.parent
			if not direction:
				result_str = " ".join(str_buffer[::-1])
			else:
				result_str = " ".join(str_buffer)
			total_score = 0
			for score in score_buffer:
				total_score += score
			
			if words[0]=="<//s>":
				result_str = " ".join(["<\\s>",result_str])
			if words[-1]=="<//s>":
				result_str = " ".join([result_str,"<\\s>"])
			result.append((result_str, total_score))
		result.sort(key=lambda x: x[0], reverse=True)
		return result[:max_limit]

	queries = error_region_scan(puzzle, error_region)
	print(queries)
	corrections = []
	for query in queries:
		words = query.split()
		if len(words) > 7:
			continue
		meta_res, start_signal, end_signal = _search(es, words)
		left_sweep_res = _sweep(words, end_signal, meta_res)
		right_sweep_res = _sweep(words, start_signal, meta_res, 1)
		if sweep_comb == "union":
			res = []
			left_dict = {text:score for text, score in left_sweep_res}
			for item in right_sweep_res:
				if item[0] not in left_dict:
					res.append((item[0], item[1]))
				else:
					res.append((item[0], (item[1] + left_dict[item[0]]) / 2))
					left_dict.pop(item[0])
			for key, value in left_dict.items():
				res.append((key, value))
			res.sort(key=lambda x: x[1], reverse=True)
		elif sweep_comb == "intersection":
			res = []
			left_dict = {text:score for text, score in left_sweep_res}
			for item in right_sweep_res:
				if item[1] in left_dict:
					res.append((item[0] + left_dict[item[1]] / 2, item[1]))
		corrections.append(res[:size])
	return corrections
				
			
		
'''	def _sweep(query, max_limit=1000, direction=0):
		result = []
		if query[0] == "<\\s>" or query[-1] == "<\\s>":
			raise NotImplementedError
		words = query.split()
		if not direction:
			new_leaves = []
			root = TreeNode((0,""))
			if words[0] == "<\\s>":
				raise NotImplementedError
			else:
				temp = TreeNode((0, words[0]))
				root.add_child(temp)
				leaves = [temp]
			for i in range(len(words) - num_of_grams + 1):
				end = False
				ngram = words[i:i + num_of_grams]
				if i == len(words) - num_of_grams:
					if ngram[-1] == "<\\s>":
						break
					else:
						end = True
				search_res = search_retrieve_wrapper(es, ngram, num_of_grams, size=max_limit)
				search_res = search_res.sort(key=lambda x: x[0], reverse=True)
				for leaf in leaves:
					for res in search_res:
						candidate = res[1].split()
						if end:
							if candidate[0] == leaf.data[-1] and candidate[-1] == ngram[-1]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
						else:
							if candidate[0] == leaf.data[-1]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
					
				leaves = new_leaves[:]
				new_leaves = []
			for leaf in leaves:
				current = leaf
				str_buffer = []
				score_buffer = []
				while current is not None:
					score_buffer.append(current, data[0])
					if current.parent is not root:
						str_buffer.append(" ".join(current.data[1].split()[1:]))
					else:
						str_buffer.append(current.data[1])
						break
					current = current.parent
				result_str = " ".join(str_buffer[::-1])
				total_score = 0
				for score in score_buffer:
					total_score += score
				result.append((result_str, total_score))
		else:
			new_leaves = []
			root = TreeNode((0,""))
			if words[-1] == "<\\s>":
				raise NotImplementedError
			else:
				temp = TreeNode((0, words[-1]))
				root.add_child(temp)
				leaves = [temp]
			for i in range(len(words) - 1, num_of_grams - 2, -1):
				end = False
				ngram = words[i-num_of_grams+1:i+1]
				if i == num_of_grams - 1:
					if ngram[0] == "<\\s>":
						break
					else:
						end = True
				search_res = search_retrieve_wrapper(es, ngram, num_of_grams)
				search_res = search_res.sort(key=lambda x: x[0], reverse=True)
				for leaf in leaves:
					for res in search_res:
						candidate = res[1].split()
						if end:
							if candidate[-1] == leaf.data[0] and candidate[0] == ngram[0]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
						else:
							if candidate[-1] == leaf.data[0]:
								new_leaf = TreeNode((res[0], res[1]))
								leaf.add_child(new_leaf)
								new_leaves.append(new_leaf)
				leaves = new_leaves[:]
				new_leaves = []
			for leaf in leaves:
				current = leaf
				str_buffer = []
				score_buffer = []
				while current is not None:
					score_buffer.append(current, data[0])
					if current.parent is not root:
						str_buffer.append(" ".join(current.data[1].split()[:-1]))
					else:
						str_buffer.append(current.data[1])
						break
					current = current.parent
				result_str = " ".join(str_buffer)
				total_score = 0
				for score in score_buffer:
					total_score += score
				result.append((result_str, total_score))
		result.sort(key=lambda x: x[0], reverse=True)
		return result[:max_limit]
'''
				
'''for i in range(len(words) - 1, num_of_grams - 2, -1):
				end = False
				ngram = words[i-num_of_grams+1:i+1]
				if i == num_of_grams - 1:
					if ngram[0] == "<\\s>":
						break
					else:
						end = True
				search_res = search_retrieve_wrapper(es, " ".join(ngram), num_of_grams, size=max_limit)
				search_res.sort(key=lambda x: x[0], reverse=True)

'''						
						
						
			

if __name__ == "__main__":
	from elasticsearch import Elasticsearch
	es = Elasticsearch()
	#text = "ok so i sould be" #able to doin any time uh what's the address of the restaurant"
	#error = [0,0,0,0,0]#,0,0,1,0,0,1,0,0,0,0,0,0]
	#print(error_region_scan(text,error))
	#with open ("correction_test.json", "w") as f:
	#	json.dump(fixed_length_algorithm(es, text, error), f, ensure_ascii=False, indent=4)
	correction_wrapper(es,"dstc10_logs_error_region_no_disfluecny.json",result_path="correction_results_7_max.json" )	
