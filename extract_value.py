import json
f=open("pos_text_summary_rating.txt","w")
from pprint import pprint
with open('pos_amazon_cell_phone_reviews.json') as data_file:    
    data = json.load(data_file)
punc=[".",',',"'",'-','_',"\"","#"]
for i in range(len(data["root"])):
	s1=""
	for j in data["root"][i]["text"]:
		for p in punc:
			j=j.replace(p,'')
		s1+=j
	data["root"][i]["text"]=s1
	#print data["root"][i]["text"]
	f.write(data["root"][i]["text"]+"#")
	s2=""
	for j in data["root"][i]["summary"]:
		for p in punc:
			j=j.replace(p,'')
		s2+=j
	data["root"][i]["summary"]=s2
	f.write(data["root"][i]["summary"]+"#")
	f.write(str(data["root"][i]["rating"]))
	f.write("\n")
f.close()