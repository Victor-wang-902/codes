from elasticsearch import Elasticsearch
import os
import json
import tqdm


def search_wrapper(es, path, filename, index_name, save=True, write_template=False):
	json_obj = dict()
	with open(os.path.join(path, filename), "r") as f:
		queries = json.load(f)
	for i, query in tqdm.tqdm(queries.items()):
		ch_sep = query["char_sequence_no_punc"]
		word_sep = query["word_sequence_no_punc"]
		result = search(es, index_name, [ch_sep, word_sep], methods="all")
		json_obj[i] = result
	if save:
		with open("search_result_new.json", "w", encoding="utf-8") as f:
			json.dump(json_obj, f, ensure_ascii=False, indent=4)

def search(es, index_name, query, methods):
	methods = ["whole_w", "char_sep", "3_gram", "3_gram_alt", "phonetic_only", "phonetic_ngram", "phonetic_ngram_alt"] if methods == "all" else methods
	result = dict()
	ch_sep, word_sep = query
	query_bodies = {"3_gram":
		{
		"size": 5,
		"query": {
			"match": {
				"3_gram": word_sep,
				}
			},
		"fields": [
			"3_gram"
			]
		},
		"whole_w":
			{
		"size": 5,
		"query": {
			"match": {
				"whole_w": word_sep,
				}
			},
		"fields": [
			"whole_w"
			]
		},
		"3_gram_alt":
			{
		"size": 5,
		"query": {
			"match": {
				"3_gram_alt": word_sep,
				}
			},
		"fields":[
			"3_gram_alt"
			]
		},
		"phonetic_ngram":
			{
		"size": 5,
		"query": {
			"match": {
				"phonetic_ngram": word_sep,
				}
			},
		"fields": [
			"phonetic_ngram"
			]
		},
		"phonetic_only":
			{
		"size": 5,
		"query": {
			"match": {
				"phonetic_only": word_sep,
				}
			},
		"fields": [
			"phonetic_only"
			]
		},
		"phonetic_ngram_alt":
			{
		"size": 5,
		"query": {
			"match": {
				"phonetic_ngram_alt": word_sep,
				}
			},
		"fields":[
			"phonetic_ngram_alt"
			]
		},
		"char_sep":
			{
		"size": 5,
		"query": {
			"match": {
				"char_sep": ch_sep,
				}
			},
		"fields":[
			"char_sep"
			]
		},
		}
	result[word_sep] = dict()
	for method in methods:		
		response = es.search(
				index=index_name,
				body = query_bodies[method])
		result[word_sep]["max_score"] = None
		result[word_sep][method] = dict()
		result[word_sep][method]["max_score"] = response["hits"]["max_score"]
		result[word_sep][method]["hits"] = []
		result[word_sep][method]["hits"] += [{"index": r["_index"], "id": r["_id"], "score": r["_score"], "result": r["_source"][method]} for r in response["hits"]["hits"]]
	return result	
	

def comparison_wrapper(es, path, filename, index_name, save=True, write_template=False):
	json_obj = dict()
	with open(os.path.join(path, filename), "r") as f:
		queries = f.readlines()
	for i, query in tqdm.tqdm(enumerate(queries)):
		result = compare(es, index_name, query.strip(), methods="all")
		json_obj[i] = result
	if save:
		with open("comparison_result.json", "w", encoding="utf-8") as f:
			json.dump(json_obj, f, ensure_ascii=False, indent=4)

def compare(es, index_name, query, methods):
	methods = ["naive_strict_trigram", "dedicated_trigram", "fuzziness", "phonetic", "phonetic_dedicated_trigram", "phonetic_fuzziness", "phonetic_dedicated_trigram_fuzziness"] if methods == "all" else methods
	result = dict()
	query_bodies = {
		#"naive_trigram": {
		#	"size": 5,
		#	"query": {
		#		"match": {
		#			"text.naive_trigram": query,
		#			}
		#		},
		#	},
		"naive_strict_trigram": {
			"size": 5,
			"query": {
				"match": {
					"text.naive_strict_trigram": query,
					}
				},
			},
		"dedicated_trigram": {
			"size": 5,
			"query": {
				"match": {
					"text.dedicated_trigram": query,
					}
				},
			},
		"phonetic": {
			"size": 5,
			"query": {
				"match": {
					"text.phonetic": query,
					}
				},
			},
		#"phonetic_naive_trigram": {
		#	"size": 5,
		#	"query": {
		#		"match": {
		#			"text.phonetic_naive_trigram": query,
		#			}
		#		},
		#	},
		#"phonetic_naive_strict_trigram": {
		#	"size": 5,
		#	"query": {
		#		"match": {
		#			"text.phonetic_naive_trigram": query,
		#			}
		#		},
		#	},
		"phonetic_dedicated_trigram": {
			"size": 5,
			"query": {
				"match": {
					"text.phonetic_dedicated_trigram": query,
					}
				},
			},
		#"phonetic_fuzziness": {
		#	"size": 5,
		#	"query": {
		#		"match": {
		#			"text.phonetic": {
		#				"query": query,
		#				"fuzziness": "AUTO",
		#				}
		#			}
		#		}
		#	},
		#"phonetic_dedicated_trigram_fuzziness": {
		#	"size": 5,
		#	"query": {
		#		"match": {
		#			"text.phonetic_dedicated_trigram": {
		#				"query": query,
		#				"fuzziness": "AUTO",
		#				}
		#			}
		#		}
		#	},
		#"default_match": {
		#	"size": 5,
		#	"query": {
		#		"match": {
		#			"text.vanilla": query,
		#			}
		#		}
		#	},
		"fuzziness": {
			"size": 5,
			"query": {
				"match": {
					"text.vanilla": { 
						"query": query, 
						"fuzziness": "AUTO",
							}
					}
				}
			}
					
		}
	result[query] = dict()
	for method in methods:		
		response = es.search(
				index=index_name,
				body = query_bodies[method])
		result[query]["max_score"] = None
		result[query][method] = dict()
		result[query][method]["max_score"] = response["hits"]["max_score"]
		result[query][method]["hits"] = []
		result[query][method]["hits"] += [{"index": r["_index"], "id": r["_id"], "score": r["_score"], "result": r["_source"]["text"]} for r in response["hits"]["hits"]]
	return result	

def search_retrieve_wrapper(es, q, num_of_grams, size=1000, method="default"):
	if method == "default":
		response = search_bigram_test(es, q, num_of_grams, size=size)
	elif method == "precision":
		response = precision_search(es, q, size=size)
	result = [(r["_score"], r["_source"]["text"]) for r in response["hits"]["hits"]]
	return result	


def precision_search(es, q, size=1000):
	min_char = 1000
	max_char = -1
	avg_char = 0
	char_len_map = []
	def _analyze_query_statistics(q):
		min_char = 1000
		max_char = -1
		avg_char = 0
		total = 0
		char_len_map = []
		for word in q.split():
			if len(word) < min_char:
				min_char = len(word)
			if len(word) > max_char:
				max_char = len(word)
			total += len(word)
			char_len_map.append(len(word))
		avg_char = total / len(q.split())
		return min_char, max_char, avg_char, char_len_map
	min_char, max_char, avg_char, char_len_map = _analyze_query_statistics(q)
	query_body = {
		"size": size,
		"query": {
			"function_score": {
				"query": {
					"bool": {
						"should": [
							{
							"match": {
								"text.naive_strict_trigram": {
									"query": q,
									"boost": 1
										}
								}
							},
							{
							"match": {
								"text.dedicated_trigram": {
									"query": q,
									"boost": 1
										}
								}
							},
							{
							"match": {
								"text.phonetic": {
									"query": q,
									"boost": 30
										}
								}
							},
							{
							"match": {
								"text.phonetic_dedicated_trigram": {
									"query": q,
									"boost": 1
										}
								}
							},
							{
							"match": {
								"text.vanilla" : {
									"query": q,
									"fuzziness": "AUTO",
									"boost": 30
										}
								}
							}
							]	
						}
					},
				#"functions": [
				#	{	
				#	"script_score": {
				#		"script": {
							#"params": {
							#	"min_char" : min_char,
							#	"max_char" : max_char,
							#	"avg_char" : avg_char,
							#	"scale_sqr": 25
							#	},
							#"source": "0.5 + Math.exp(-(Math.max(0, (Math.sqrt((Math.pow((doc['statistics.min_char'] - params.min_char), 2) + Math.pow((doc['statistics.max_char'] - params.max_char), 2) + Math.pow((doc['statistics.avg_char'] - params.avg_char), 2)) / 3))) / (2 * (- params.scale_sqr / (2 * Math.ln(0.5))))))",
				#			"source": "1",
				#			}
				#		},
				#	"weight": 1
				#	},
					#{
					"script_score": {
						"script": {
							#"params": {
							#	"freq": "doc['statistics.freq'].value"
							#	},
							#"source": "1 + 0.1 * Math.log10(9 + params.freq)"
							"source": "1"
							}	
						},
					#	"weight": 1, 
				#	],
			"max_boost": 1000,
			"score_mode": "avg",
			"boost_mode": "multiply",
			"min_score": 1
				}
			}
		}
	result = es.search(index="1_gram", body=query_body)
	return result

def search_bigram_test(es, q, num_of_grams=2, size=1000):
	index_mapping = {1: "1_gram", 2: "2_gram", 3: "3_gram", 4: "4_gram", 5: "5_gram", 6: "6_gram"}
	index_name = index_mapping[num_of_grams]
	min_char = 1000
	max_char = -1
	avg_char = 0
	char_len_map = []
	def _analyze_query_statistics(q):
		min_char = 1000
		max_char = -1
		avg_char = 0
		total = 0
		char_len_map = []
		for word in q.split():
			if len(word) < min_char:
				min_char = len(word)
			if len(word) > max_char:
				max_char = len(word)
			total += len(word)
			char_len_map.append(len(word))
		avg_char = total / len(q.split())
		return min_char, max_char, avg_char, char_len_map
	min_char, max_char, avg_char, char_len_map = _analyze_query_statistics(q)
	query_body = {
		"size": size,
		"query": {
			"function_score": {
				"query": {
					"bool": {
						"should": [
							{
							"match": {
								"text.naive_strict_trigram": {
									"query": q,
									"boost": 10
										}
								}
							},
							{
							"match": {
								"text.dedicated_trigram": {
									"query": q,
									"boost": 10
										}
								}
							},
							{
							"match": {
								"text.phonetic": {
									"query": q,
									"boost": 30
										}
								}
							},
							{
							"match": {
								"text.phonetic_dedicated_trigram": {
									"query": q,
									"boost": 20
										}
								}
							},
							{
							"match": {
								"text.vanilla" : {
									"query": q,
									"fuzziness": "AUTO",
									"boost": 30
										}
								}
							}
							]	
						}
					},
				#"functions": [
				#	{	
				#	"script_score": {
				#		"script": {
							#"params": {
							#	"min_char" : min_char,
							#	"max_char" : max_char,
							#	"avg_char" : avg_char,
							#	"scale_sqr": 25
							#	},
							#"source": "0.5 + Math.exp(-(Math.max(0, (Math.sqrt((Math.pow((doc['statistics.min_char'] - params.min_char), 2) + Math.pow((doc['statistics.max_char'] - params.max_char), 2) + Math.pow((doc['statistics.avg_char'] - params.avg_char), 2)) / 3))) / (2 * (- params.scale_sqr / (2 * Math.ln(0.5))))))",
				#			"source": "1",
				#			}
				#		},
				#	"weight": 1
				#	},
					#{
					"script_score": {
						"script": {
							#"params": {
							#	"freq": "doc['statistics.freq'].value"
							#	},
							#"source": "1 + 0.1 * Math.log10(9 + params.freq)"
							"source": "1"
							}	
						},
					#	"weight": 1, 
				#	],
			"max_boost": 1000,
			"score_mode": "avg",
			"boost_mode": "multiply",
			"min_score": 1
				}
			}
		}
	result = es.search(index=index_name, body=query_body)
	return result		


if __name__ ==  "__main__":
	es = Elasticsearch()
	queries = [
		("/scratch/zw2374/public/es_exp/",
		"bigram_puzzles.txt")
		]
	#index_name = "dstc10_train_repunc_no_disfluencies"
	#for query in queries:
	#	search_wrapper(es, query[0], query[1], index_name)
	rs = search_retrieve_wrapper(es, "code d", 2, 50)
	with open("trial_bigram.json", "w") as f:
		json.dump(rs, f, ensure_ascii=False, indent=4)
	#index_name = "bigram_test"
	#comparison_wrapper(es, queries[0][0], queries[0][1], "bigram_test")
