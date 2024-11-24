import os
import sys
import xml.etree.ElementTree as ET
import operator
from nltk.stem.snowball import SnowballStemmer
import json
import random
stupid_tags = [ '-', '*','wikipedia','wiki','interesting','article' ]

def get_categories(root):
    category = {}
    idx = 1
    for child in root:
        try:
            title = child[1].text
            identifier = child[0].text
            for tag_par in child[3]:
                tag = tag_par[0].text
                count = int(tag_par[1].text)
                # Both the tag and count may be switched around,
                # swap to have correctness

                if tag in stupid_tags:
                    continue
                if tag in category and not tag.isdigit():
                    category[tag] += 1
                else:
                    category[tag] = 1
            idx += 1
        except:
            pass
    return category

def make_category_for_doc(root, category):
    idx = 1
    for child in root:
        try:
            title = child[1].text
            identifier = child[0].text
            chosen_tag = None
            # tags are in sorted order
            # hence we iterate from top tag to less relevent tag automatically
            for tag_par in child[3]:
                tag = tag_par[0].text
                count = tag_par[1].text


                # Most relevent tag present in our category set is chosen
                if tag in category and tag not in stupid_tags:
                    chosen_tag = tag
            # If there is no chosen tag, it means that the document needs to be discarded
            if not chosen_tag or chosen_tag in stupid_tags:
                continue
            print (idx, title, identifier, chosen_tag)
            idx += 1
        except:
            pass

def extract_tag(file_path,number):
    tag_tree = ET.parse(file_path)
    root = tag_tree.getroot()
    category = get_categories(root)
    sorted_category = sorted(category.items(), key=operator.itemgetter(1),reverse=True)
    category_set =  dict(sorted_category[:number])
    
    make_category_for_doc(root, category_set)
    print(category_set)

def build_new_tag_for_25actegory(file_path='wiki10+_tag-data/tag-data.xml',train_out_filepath='new_dataset_for_25_top_cat_train.json',test_out_filepath='new_dataset_for_25_top_cat_test.json',choosen_cate_number=25):
    tag_tree = ET.parse(file_path)
    root = tag_tree.getroot()
    category = get_categories(root)
    sorted_category = sorted(category.items(), key=operator.itemgetter(1),reverse=True)
    category_set =  dict(sorted_category[:choosen_cate_number])
    train_dataset={}
    test_dataset={}
    train_dataset['categories']= list(category_set.keys())
    test_dataset['categories']= list(category_set.keys())
    hash_categories_pair=[]
    # the newdataset will be stored in a json file. a big dictionary,['hash',title,[category1,category2..]]
    for child in root:
        try:
            title = child[1].text
            identifier = child[0].text
            categories=[]

            for tag_par in child[3]:
                if tag_par[0].text in category_set:
                    categories.append(tag_par[0].text)
            if categories:
                hash_categories_pair.append([identifier,title,categories])

        except:
            pass
    
    
    random.shuffle(hash_categories_pair)

    hash_categories_pair=hash_categories_pair[:120]


    
    l=len(hash_categories_pair)
    train_dataset['hash_categories_pair']=hash_categories_pair[:int(2*l/3)]
    test_dataset['hash_categories_pair']=hash_categories_pair[int(2*l/3):]
    with open(train_out_filepath, "w") as json_file:
        json.dump(train_dataset, json_file, indent=4)  # indent 参数用于美化输出，设置为4可以让 JSON 更易读
    with open(test_out_filepath, "w") as json_file:
        json.dump(test_dataset, json_file, indent=4)  # indent 参数用于美化输出，设置为4可以让 JSON 更易读




if __name__ == "__main__":

    build_new_tag_for_25actegory('wiki10+_tag-data/tag-data.xml',choosen_cate_number=25)
    
