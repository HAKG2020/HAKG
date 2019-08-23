from GRUModel import GGNN
from IGRUTrain import IGRUTrain
from data_preprocess import *
import numpy as np
import argparse
import torch
import json

import pickle
import random
from datetime import datetime

def load_file_as_dict(filename):
    dict = {}
    with open(filename, "r") as file:
        for line in file:
            arr = line.replace('\n','').replace('\r','').split("\t")
            userid, itemid = arr[0], arr[1]
            if userid not in dict:
                dict.update({userid:[itemid]})
            else:
                dict[userid].append(itemid)
    return dict

def get_negative_label(fr_file):
    negative_label = []
    for line in fr_file:
        lines = line.replace('\n', '').replace('\r','').split('\t')
        user = 'u' + lines[0]
        item = 'i' + lines[1]
        key = (user, item)
        negative_label.append(key)
    return negative_label


def get_positive_label(fr_file):
    positive_label = []
    for line in fr_file:
        lines = line.replace('\n', '').replace('\r','').split('\t')
        user = 'u' + lines[0]
        item = 'i' + lines[1]
        key = (user, item)
        positive_label.append(key)
    return positive_label


def load_paths(fr_file):
    paths_between_pairs = {}
    for line in fr_file:
        nodes_in_a_path = line.replace('\n', '').replace('\r','').split(',')
        user_node= nodes_in_a_path[0]
        last_node = nodes_in_a_path[-1]
        if last_node == 'None':
            movie_node = nodes_in_a_path[1]
            key = (user_node, movie_node)
            paths_between_pairs.update({key: []})

        else:
            movie_node = nodes_in_a_path[-1]
            key = (user_node, movie_node)
            path = nodes_in_a_path
            if key not in paths_between_pairs:
                paths_between_pairs.update({key: [path]})
            else:
                if len(paths_between_pairs[key]) <30:
                	paths_between_pairs[key].append(path)

    return paths_between_pairs



if __name__ == "__main__":


    # learning-rate : 0.001 dimension:16-4
    parser = argparse.ArgumentParser()
    type_dim=32
    rel_dim = 32
    entity_dim=128
    iteration=15
    learning_rate=0.002
    weight_decay=1e-4
    step=2
    batch_size=256  # u-u, i-i, u-i ,i-g
    relation_num = 4  # user-business user-user business-city business-category
    type_num = 3  # user business city catrgory
    device_num= 4

    # res file
    train_file_name = '../ml/training.txt'
    negative_file_name ='../ml/negative.txt'
    test_negative_file_name = '../ml/negative.txt'

    # path file
    positive_path_file_name = '../ml/dag_positive.txt'
    negative_path_file_name = '../ml/dag_negative.txt'
    test_negative_path_file_name = '../ml/dag_test_negative.txt'

    #all nodes and id path
    pre_embedding_path ='../ml/embedding_ml.vec.json'


    all_nodes_and_ids_path ='../ml/all_nodes_and_ids.pickle'

    # distance parameters
    alpha = 0.1
    # set gpu
    torch.cuda.set_device(device_num)


    # set random seed
    manualSeed = random.randint(1, 10000)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manualSeed)

    #train user_item pair
    positive_user_items_dict = load_file_as_dict(train_file_name)
    negative_user_items_dict = load_file_as_dict(negative_file_name)
    all_label_list = []
    for user_id in positive_user_items_dict:
        user_node = 'u' + user_id
        for item_id in positive_user_items_dict[user_id]:
            movie_node = 'i' + item_id
            all_label_list.append((user_node,movie_node))
        for item_id in negative_user_items_dict[user_id]:
            movie_node = 'i' + item_id
            all_label_list.append((user_node,movie_node))


    with open(all_nodes_and_ids_path, 'rb') as file:
          all_nodes_and_ids = pickle.load(file)

    node_size = len(all_nodes_and_ids)
    start_time = datetime.now()

    # load pre_embedding
    entity_pre_embedding = np.random.rand(node_size + 1, entity_dim)  # embeddings for all nodes
    with open(pre_embedding_path,'rb') as file:
         embedding_dict = json.load(file)
    rel_embedding_list = embedding_dict['rel_embeddings']
    ent_embedding_list = embedding_dict['ent_embeddings']
    print(len(ent_embedding_list))
    for i in range(node_size):
        entity_pre_embedding[i] = np.array(ent_embedding_list[i])
    rel_pre_embedding = np.array(rel_embedding_list)
    print('load_pre_embedding finished, two embedding matrix is ',entity_pre_embedding.shape,rel_pre_embedding.shape)
    entity_pre_embedding = torch.FloatTensor(entity_pre_embedding)
    rel_pre_embedding = torch.FloatTensor(rel_pre_embedding)


    # prepare  paths

    positive_path_dict = readAllSequencesFromFile(positive_path_file_name)
    print('load positive finished')
    negative_path_dict = readAllSequencesFromFile(negative_path_file_name)
    print('load negative finished')
    test_negative_path_dict = readAllSequencesFromFile(test_negative_path_file_name)
    #test_negative_path_dict = positive_path_dict
    print('load test_negative finished')
    load_end_time = datetime.now()
    duration = load_end_time-start_time
    print('load all data finished, time is  :  ', duration)

    # get model instance

    model_relation = GGNN(entity_dim, type_dim, rel_dim, node_size, type_num, relation_num, entity_pre_embedding)
    #model_relation = torch.load('ml_dag/model_epoch_5.pt')
    print(model_relation)
    model_relation.double()  # important!! cannot delete

    if torch.cuda.is_available():
        model_relation = model_relation.cuda()
        print('model_to_cuda,single-gpu')

    # train and test the model

    model_relation.train()

    model_trained_relation = IGRUTrain(model_relation, iteration, learning_rate, weight_decay,
                 all_label_list, all_nodes_and_ids, positive_path_dict,
                 negative_path_dict, test_negative_path_dict,type_num,relation_num,alpha)
    model_trained_relation.train_relation()

    print('model training and testing finished')
