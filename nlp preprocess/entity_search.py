from elasticsearch import Elasticsearch
import os
from extra_methods import convert_text2num
import json
import tqdm
from kb_index import create_kb, bulk_index_temp


def entity_retrieve_wrapper(es, q, area, size=10, entity_dict=None, index_name="kb_only"):
	mapp = {"hotel" : 100000000, "restaurant": 200000000, "attraction": 300000000}
	print(q)
	q = convert_text2num(q)
	response = entity_search(es, q, area, entity_dict, index_name, size=size)
	result = [(r["_score"], r["_source"]["category"] + "-" + str(int(r["_id"]) - mapp[r["_source"]["category"]]), r["_source"]["text"],) for r in response["hits"]["hits"]]
	return result	

def entity_search(es, q, area, entity_dict, index_name, size=10):
	if not len(area):
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
									"boost": 20
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
									"boost": 30
										}
								}
							},
							{"match": {
								"text.phonetic_naive_strict_trigram": {
									"query": q,
									"boost": 50
										}
								}
							},
							{
							"match": {
								"text.vanilla" : {
									"query": q,
									"fuzziness": "AUTO",
									"boost": 25
										}
								}
							},
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
	else:
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
									"boost": 20
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
									"boost": 30
										}
								}
							},
							{"match": {
								"text.phonetic_naive_strict_trigram": {
									"query": q,
									"boost": 50
										}
								}
							},
							{
							"match": {
								"text.vanilla" : {
									"query": q,
									"fuzziness": "AUTO",
									"boost": 25
										}
								}
							},
							{
							"bool": {
								"should": [
										{
										"match": {
											"area": {
											"query": a,
											"fuzziness": "AUTO",
											"boost": 25
											}
										}
									} for a in area.values()
								]
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
	if entity_dict:
		list_of_entity = [item[0] for item in entity_dict.values()]
		print(list_of_entity)
		query_body["query"]["function_score"]["query"]["bool"]["must"] = [{"bool": {"should": [{ "match": { "text.vanilla": { "query": entity, "boost": 30 } } }  for entity in list_of_entity]}}]
		print(query_body)
	result = es.search(index=index_name, body=query_body)
	return result		
def search_system_no_area(es,input_path, output_path):
	with open(input_path, "r") as f:
		obj = json.load(f)
	for dialog in obj:
		for turn in dialog:
			if turn["speaker"] == "S":
				turn["id_map_es_new"] = {}
				for entity in turn["entity_3"]:
					temp = entity_retrieve_wrapper(es,entity,{},1,None,index_name="kb_only")
					for t in temp:
						turn["id_map_es_new"]["<" + t[1] + ">"] = entity
	with open(output_path, "w") as f:
		json.dump(obj, f, ensure_ascii=False, indent=4)	


def search_system(es,input_path, output_path):
	with open(input_path, "r") as f:
		obj = json.load(f)
	for dialog in obj:
		for turn in dialog:
			if turn["speaker"] == "S":
				turn["id_map_es_new"] = {}
				try:
					area = turn["area_map"]
				except KeyError:
					area = turn["area"]
				for entity in turn["entity_3"]:
					temp = entity_retrieve_wrapper(es,entity,area,1,None,index_name="kb_only")
					for t in temp:
						turn["id_map_es_new"]["<" + t[1] + ">"] = entity
	with open(output_path, "w") as f:
		json.dump(obj, f, ensure_ascii=False, indent=4)	

def search_user(es,input_path, output_path):
	with open(input_path, "r") as f:
		obj = json.load(f)
	for dialog in obj:
		for turn in dialog:
			if turn["speaker"] == "U":
				turn["id_map2_es_new"] = {}
				for entity in turn["entity_3"]:
					temp = entity_retrieve_wrapper(es,entity,area,10,None, index_name="kb_only")
					print(temp)
					for t in temp:
						turn["id_map2_es_new"]["<" + t[1] + ">"] = entity
	with open(output_path, "w") as f:
		json.dump(obj, f, ensure_ascii=False, indent=4)	

def select_user_no_area(es,input_path, output_path, entity_path="knowledge_entity.json"):
	with open(input_path, "r") as f:
		obj = json.load(f)
	for dialog in obj:
		for turn in dialog:
			if turn["speaker"]== "U":
				thingy = turn["candidate_dict"]
				break
		bulk_index_temp(es, thingy)
		for turn in dialog:
			if turn["speaker"] == "U":
				turn["id_map_es_new"] = {}
				for entity in turn["entity_3"]:
					temp = entity_retrieve_wrapper(es, entity,{}, 1, None, index_name="temp")
					for t in temp:
						turn["id_map_es_new"]["<"+ t[1] + ">"] = entity
	with open(output_path, "w") as f:
		json.dump(obj,f,ensure_ascii=False, indent=4)


def select_user(es,input_path, output_path, entity_path="knowledge_entity.json"):
	with open(input_path, "r") as f:
		obj = json.load(f)
	for dialog in obj:
		for turn in dialog:
			if turn["speaker"]== "U":
				thingy = turn["candidate_dict"]
				break
		bulk_index_temp(es, thingy)
		for turn in dialog:
			if turn["speaker"] == "U":
				try:
					area = turn["area_map"]
				except KeyError:
					area = turn["area"]
				turn["id_map_es_new"] = {}
				for entity in turn["entity_3"]:
					temp = entity_retrieve_wrapper(es, entity,area, 1, None, index_name="temp")
					for t in temp:
						turn["id_map_es_new"]["<"+ t[1] + ">"] = entity
	with open(output_path, "w") as f:
		json.dump(obj,f,ensure_ascii=False, indent=4)


if __name__ ==  "__main__":
	es = Elasticsearch()
	#rs = entity_retrieve_wrapper(es, "Peking Restaurant", 10)
	#with open("trial_entity_search_single.json", "w") as f:
#		json.dump(rs, f, ensure_ascii=False, indent=4)
	#system_paths = {"dstc10_system.json": "dstc10_system_new.json", "exclude_system.json":"exclude_system_new.json"}
	#for inn , out in system_paths.items():
	#	search_system(es, inn, out)
	#user_path = {"dstc10_user.json": "dstc10_user_new.json", "exclude_user.json": "exclude_user_new.json"}
	#for inn, out in user_path.items():
	#	search_user(es, inn, out)
	#select_path = {"dstc10_user_new.json": "dstc10_final.json", "exclude_user_new.json": "exclude_final.json"}
	#for inn, out in select_path.items():
	#	select_user(es, inn, out)
	#paths_system_no_area = {"final_data/dstc10_val_no_area_system.json": "/scratch/jx1038/public/dstc10_val_system_with_area.json", "final_data/train_no_area_system.json": "/scratch/jx1038/public/exclude_clean_version.json", "final_data/test_no_area_system.json": "/scratch/jx1038/public/user_final.json"}
	#for o,i in paths_system_no_area.items():
	#	search_system_no_area(es, i, o)
	#paths_system_with_area = {"final_data/dstc10_val_with_area_system.json": "/scratch/jx1038/public/dstc10_val_system_with_area.json", "final_data/train_with_area_system.json": "/scratch/jx1038/public/exclude_clean_version.json", "final_data/test_with_area_system.json": "/scratch/jx1038/public/user_final.json"}
	#for o, i in paths_system_with_area.items():
	#	search_system(es,i,o)
	#paths_user_no_area = {"final_data/dstc10_val_no_area_final.json": "/scratch/jx1038/public/result/dstc10_val_with_area_user.json", "final_data/train_no_area_final.json": "/scratch/jx1038/public/result/train_with_area_user.json", "final_data/test_no_area_final.json": "/scratch/jx1038/public/result/test_with_area_user.json"}
	#for o,i in paths_user_no_area.items():
	#	select_user_no_area(es, i, o)	
	paths_user_with_area = {"final_data/dstc10_val_with_area_final.json": "/scratch/jx1038/public/result/dstc10_val_with_area_user.json", "final_data/train_with_area_final.json": "/scratch/jx1038/public/result/train_with_area_user.json", "final_data/test_with_area_final.json": "/scratch/jx1038/public/result/test_with_area_user.json"}
	for o, i in paths_user_with_area.items():
		select_user(es,i,o)
	

	#select_user(es, "final_user.json", "test_final.json")	
	#search_system_no_area(es, "/scratch/jx1038/public/dstc10_val_final_user.json", "val_no_area.json")
#	select_user_no_area(es, "final_user.json", "test_final.json")
