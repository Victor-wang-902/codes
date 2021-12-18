from elasticsearch import Elasticsearch, helpers
import os
import json
import time
import tqdm


def create_bigram_trial(es, num_of_grams, index_name="bigram_test", override=False):
	request_body = {
			"settings": {
			"index": {
				"max_ngram_diff": "100",
				"analysis": {
					"analyzer": {
						"vanilla": {
							"tokenizer": "whitespace",
							"char_filter": [
								"transform"
								],
							"filter": [
								"asciifolding"
								]
							},
						#"naive_trigram": {
						#	"tokenizer": "whitespace",
						#	"char_filter": [
						#		"transform"
						#		],
						#	"filter": [
						#		"asciifolding",
						#		"naive_trigram"
						#		]
						#	},
						"naive_strict_trigram": {
							"tokenizer": "whitespace",
							"char_filter": [
								"transform",
								],
							"filter": [
								"asciifolding",
								"naive_strict_trigram"
								]
							},
						"dedicated_trigram": {
							"tokenizer": "dedicated_trigram",
							"char_filter": [
								"transform"
								],
							"filter": [
								"asciifolding",
								]
							},
						"phonetic": {
							"tokenizer": "whitespace",
							"char_filter": [
								"transform"
								],
							"filter": [
								"asciifolding",
								"phonetic"
								],
							},
						#"phonetic_naive_trigram": {
						#	"tokenizer": "whitespace",
						#	"char_filter": [
						#		"transform"
						#		],
						#	"filter": [
						#		"asciifolding",
						#		"naive_trigram",
						#		"phonetic"
						#		]
						#	},
						"phonetic_naive_strict_trigram": {
							"tokenizer": "whitespace",
							"char_filter": [
								"transform"
								],
							"filter": [
								"asciifolding",
								"naive_strict_trigram",
								"phonetic"
								]
							},
						"phonetic_dedicated_trigram": {
							"tokenizer": "dedicated_trigram",
							"char_filter": [
								"transform"
								],
							"filter": [
								"asciifolding",
								"phonetic"
								]
							},
						},	
					"char_filter": {
						"transform": {
							"type": "mapping",
							"mappings": [
							"@ => _at_",
							"# => _number_",
							"$ => _dollar_",
							"% => _percent_",
							"& => _and_",
							"/ => _slash_",
							"* => _times_",
							"+ => _plus_",
							]
						}
					},
					"filter": {
						"naive_trigram": {
							"type": "ngram",
							"min_gram": "1",
							"max_gram": "3"
						},
						"naive_strict_trigram": {
							"type": "ngram",
							"min_gram": "3",
							"max_gram": "3"
						},
						"phonetic": {
							"type": "phonetic",
							"encoder": "double_metaphone",
							"replace": "true",
							"max_code_len": 100
						},
					},
					"tokenizer": {
						"dedicated_trigram": {
							"type": "ngram",
							"min_gram": 3,
							"max_gram": 3,
							"token_chars": [
								"letter",
								"whitespace",
								"punctuation"
								]
							},

						}
					}
				}
			},
			"mappings": {
				"properties": {
					"text": {
						"type": "text",
						"index": "true",
						"fields": {
							"vanilla": {
								"type": "text",
								"analyzer": "vanilla",
								},
							#"naive_trigram": {
							#	"type": "text",
							#	"analyzer": "naive_trigram"
							#	},
							"naive_strict_trigram": {
								"type": "text",
								"analyzer": "naive_strict_trigram"
								},
							"dedicated_trigram": {
								"type":	"text",	
								"analyzer": "dedicated_trigram"
								},
							"phonetic": {
								"type": "text",
								"analyzer": "phonetic"
								},
							#"phonetic_naive_trigram": {
							#	"type": "text",
							#	"analyzer": "phonetic_naive_trigram"
							#	},
							"phonetic_naive_strict_trigram": {
								"type": "text",
								"analyzer": "phonetic_naive_strict_trigram"
								},
							"phonetic_dedicated_trigram": {
								"type": "text",
								"analyzer": "phonetic_dedicated_trigram"
								}
							}
						},
						"freq": {
							"type": "integer",
							"index": "false"
							},
						"min_char": {
							"type": "integer",
							"index": "false"
							},
						"max_char": {
							"type": "integer",
							"index": "false"
							},
						"avg_char": {
							"type": "float",
							"index": "false"
							},
						"len_map": { 
							"type": "nested",
							"properties": {
								str(i): {
									"type": "integer",
									"index": "false"
									} 
								for i in range(1, num_of_grams+1)
								}
							}
						}
					}					
				}
	print("creating index...")
	if override:
		if es.indices.exists(index=index_name):
			es.indices.delete(index=index_name)
		es.indices.create(index=index_name, body=request_body)
	else:
		es.indices.create(index=index_name, body=request_body)
					
							
								

def create_comparison_index(es, ignore=False):
	request_body = {
			"settings": {
			"index": {
				"max_ngram_diff": "100",
				"analysis": {
					"char_filter": {
						"transform": {
							"type": "mapping",
							"mappings": [
							"@ => _at_",
							"# => _number_",
							"$ => _dollar_",
							"% => _percent_",
							"& => _and_",
							"/ => _slash_",
							"* => _times_",
							"+ => _plus_",
							"<\\s> => N"
								]
							}
						},
					"analyzer": {
						"whole_word": {
							"tokenizer": "whitespace",
							"filter": [
								"transform",
								"asciifolding",
								]
							},
						"3_gram": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"3gram",
								]
							},
						"phonetic_anlz": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"db_metaphone",
								]
							},
						"phonetic_ngram": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"3gram",
								"my_metaphone",
								]
							},
						"phonetic_ngram_alt": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"3gram_max",
								"my_metaphone",
								]
							},
						"3gram_alt": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"3gram_max",
								]
							},
						},
					"filter": {
						"3gram": {
							"type": "ngram",
							"min_gram": 3,
							"max_gram": 3,
							},
						"db_metaphone": {
								"type": "phonetic",
								"encoder": "double_metaphone",
								"replace": True,
								"max_code_len": 100,
								},
						"my_metaphone": {
								"type": "phonetic",
								"encoder": "metaphone",
								"replace": True
								},
						"3gram_max": {
							"type": "ngram",
							"min_gram": 1,
							"max_gram": 3
							}
						},
					}
				}
			},
		"mappings": {
			"properties": {
				"char_sep": {"type": "text", "analyzer": "ch_sep"},
				"whole_w": {"type": "text", "analyzer": "whole_word"},
				"3_gram": {"type": "text", "analyzer": "3_gram"},
				"phonetic_only": {"type": "text", "analyzer": "phonetic_anlz"},
				"phonetic_ngram": {"type": "text", "analyzer": "phonetic_ngram"},
				"phonetic_ngram_alt": {"type": "text", "analyzer": "phonetic_ngram_alt"},
				"3_gram_alt": {"type": "text", "analyzer": "3gram_alt"}
				}
			}
		}
	print("creating index...")
	if ignore:
		es.indices.create(index=index_name, body=request_body, ignore=400)
	else:
		es.indices.create(index=index_name, body=request_body)



def create_index(es, index_name, ignore=False):
	request_body = {
		"settings": {
			"index": {
				"max_ngram_diff": "100",
				"analysis": {
					"analyzer": {
						"ch_sep": {
							"tokenizer": "whitespace",
							"filter": [
								"asciifolding",
								]
							},
						"whole_word": {
							"tokenizer": "whitespace",
							"filter": [
								"asciifolding",
								]
							},
						"3_gram": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"3gram",
								]
							},
						"phonetic_anlz": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"db_metaphone",
								]
							},
						"phonetic_ngram": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"3gram",
								"my_metaphone",
								]
							},
						"phonetic_ngram_alt": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"3gram_max",
								"my_metaphone",
								]
							},
						"3gram_alt": {
							"tokenizer": "whitespace",
							"filter": ["rm_special_tokens",
								"asciifolding",
								"3gram_max",
								]
							},
						},
					"filter": {
						"rm_special_tokens": {
								"type": "stop",
								"stopwords": [ "<hotel>", "<restaurant>", "<num>", "<dig>", "attraction>", ]
	
								},
						"3gram": {
							"type": "ngram",
							"min_gram": 3,
							"max_gram": 3,
							},
						"db_metaphone": {
								"type": "phonetic",
								"encoder": "double_metaphone",
								"replace": True,
								"max_code_len": 100,
								},
						"my_metaphone": {
								"type": "phonetic",
								"encoder": "metaphone",
								"replace": True
								},
						"3gram_max": {
							"type": "ngram",
							"min_gram": 1,
							"max_gram": 3
							}
						},
					}
				}
			},
		"mappings": {
			"properties": {
				"char_sep": {"type": "text", "analyzer": "ch_sep"},
				"whole_w": {"type": "text", "analyzer": "whole_word"},
				"3_gram": {"type": "text", "analyzer": "3_gram"},
				"phonetic_only": {"type": "text", "analyzer": "phonetic_anlz"},
				"phonetic_ngram": {"type": "text", "analyzer": "phonetic_ngram"},
				"phonetic_ngram_alt": {"type": "text", "analyzer": "phonetic_ngram_alt"},
				"3_gram_alt": {"type": "text", "analyzer": "3gram_alt"}
				}
			}
		}
	print("creating index...")
	if ignore:
		es.indices.create(index=index_name, body=request_body, ignore=400)
	else:
		es.indices.create(index=index_name, body=request_body)

def bulk_index(es, index_name, data_path):
	def gendata():
		time.sleep(1)
		count = es.count(index=index_name)
		count = count["count"]
		print(count)
		with open(data_path, "r") as f:
			obj = json.load(f)
		for key, value in obj.items():
			yield {
				"_index": index_name,
				"_id": int(key) + count,
				"_source": {
					"char_sep": value["char_sequence_repunc_without_punctuation"],
					"whole_w": value["word_sequence_repunc_without_punctuation"],
					"3_gram": value["word_sequence_repunc_without_punctuation"],
					"phonetic_only": value["word_sequence_repunc_without_punctuation"],
					"phonetic_ngram": value["word_sequence_repunc_without_punctuation"],
					"phonetic_ngram_alt": value["word_sequence_repunc_without_punctuation"],
					"3_gram_alt": value["word_sequence_repunc_without_punctuation"]
					}
				}
	print(helpers.bulk(es, gendata()))

def bulk_index_bigrams_trial(es, num_of_grams, data_path, index_name="bigram_test"):
	def gendata():
		time.sleep(1)
		count = es.count(index=index_name)
		count = count["count"]
		with open(data_path, "r") as f:
			obj = json.load(f)
		for i, (doc, params) in enumerate(tqdm.tqdm(obj.items())):
			yield {
				"_index": index_name,
				"_id": i,
				"_source": {
					"text" : doc,
					"freq": params["freq"],
					"min_char": params["min_char"],
					"max_char": params["max_char"],
					"avg_char": params["avg_char"],
					"len_map": { str(j): 
							params["len_map"][j-1] 
							for j in range(1, num_of_grams + 1)
						}	
					}
				}
	helpers.bulk(es, gendata())

					

def main():
	es = Elasticsearch()
#	index_name = "dstc10_train_repunc_no_disfluencies"
#	create_index(es, index_name, ignore=True)
#	to_index = ["DSTC9_task1_kb_eval_templates_es_sequences_repunc_dis.json", "DSTC9_task1_kb_templates_es_sequences_repunc_dis.json", "DSTC9_task1_train_logs_templates_es_sequences_repunc_dis.json", "DSTC9_task1_val_log_templates_es_sequences_repunc_dis.json"]
#	for item in to_index:
#		bulk_index(es, index_name, item)
	for i in range(1,7):
		create_bigram_trial(es, i, index_name=str(i) + "_gram", override=True)
		bulk_index_bigrams_trial(es, i, data_path= str(i) + "_grams_map.json", index_name=str(i) + "_gram")

main()
	
			
				
						
					
						
						
