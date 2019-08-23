import networkx as nx
import pickle
import json
import numpy as np

import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import numpy
# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def discountForPathlength(beta, length):
    return numpy.exp(-beta*length)

def readAllSequencesFromFile(dependencySaveFile):
    map={}
    with open(dependencySaveFile) as f:
        for l in f:
            tmp=l.strip().split('&')


            #print(tmp[2]) dependency
            user = tmp[0].split('-')[0]
            item = tmp[0].split('-')[1]
            key = (user,item)
            if key not in map:
                map[key]=[]
            maxLen=0
            sequence=[]

            pathsTmp=tmp[1].replace('#','\t')
            paths=pathsTmp.strip().split('\t')


            if paths[0] == 'None-0':

                dependenciesList=[]
                arr = [sequence, dependenciesList, 0]
                map[key].append(arr)
                continue


            for path in paths:
                pathList=[]
                nodes=path.strip().split(' ')
                for node in nodes:
                    ns=node.strip().split('-')

                    id0=ns[0] # node id
                    id1=int(ns[1])
                    pathList.append([id0, id1])
                    if id1>maxLen:
                        maxLen=id1

                sequence.append(pathList)

            if len(tmp[2])==0:
                dependenciesList=[]
                arr = [sequence, dependenciesList, maxLen + 1]
                map[key].append(arr)
                continue
            dependencies=tmp[2].strip().split(' ')
            dependenciesList=[]
            for depend in dependencies:
                dep=depend.strip().split('<-')
                depList=[]
                d0=dep[0].strip().split(':')
                d1=dep[1].strip().split(':')
                depList.append(d0[0])
                depList.append(d0[1])
                depList.append(d1[0])
                depList.append(d1[1])
                dependenciesList.append(depList)

            arr = [sequence, dependenciesList, maxLen + 1]

            # maxLen + 1 : node_num in the sequence
            map[key].append(arr)

    f.close()
    f=None
    return map
def load_file_as_dict(filename):
    dict = {}
    with open(filename, "r") as f:
        for line in f:
            arr = line.replace('\n', '').replace('\r', '').split("\t")
            userid, itemid = arr[0], arr[1]
            if userid not in dict:
                dict.update({userid: [itemid]})
            else:
                dict[userid].append(itemid)
    return dict


# [ [user,item],[user,iem]]
def load_test_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.replace('\n', '').replace('\r', '').split("\t")
            userid, itemid = arr[0], arr[1]
            ratingList.append([userid, itemid])
            line = f.readline()
    return ratingList


# [ [i1,i2,i3],[i3,i5,i5]]
def load_test_negative_file_as_list(filename):
    negativeList = []
    test_Ratings = []
    with open(filename, "r") as f:
        for line in f:
            arr = line.replace('\n', '').replace('\r', '').split("\t")
            userid, itemid = arr[0], arr[1]
            test_Ratings.append([userid, itemid])
            negatives = []
            for x in arr[2:]:
                negatives.append(x)
            negativeList.append(negatives)

    print(test_Ratings)
    print('num of test users', len(test_Ratings), len(negativeList))

    return test_Ratings, negativeList


def getP(ranklist, gtItems):
    p = 0
    for item in ranklist:
        if item in gtItems:
            p += 1
    return p * 1.0 / len(ranklist)


def getR(ranklist, gtItems):
    r = 0
    for item in ranklist:
        if item in gtItems:
            r += 1
    return r * 1.0 / len(gtItems)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item in gtItem:
            return 1
    return 0


def getDCG(ranklist, gtItems):
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in gtItems:
            dcg += 1.0 / math.log(i + 2)
    return dcg


def getIDCG(ranklist, gtItems):
    idcg = 0.0
    i = 0
    for item in ranklist:
        if item in gtItems:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg


def getNDCG(ranklist, gtItems):
    dcg = getDCG(ranklist, gtItems)
    idcg = getIDCG(ranklist, gtItems)
    if idcg == 0:
        return 0
    return dcg / idcg


def generate_batch(length, batch_size, shuffle=True):
    n_batch = int(length / batch_size)
    if length % batch_size != 0:
        n_batch += 1
    slices = np.split(np.arange(n_batch * batch_size), n_batch)
    slices[-1] = slices[-1][:(length - batch_size * (n_batch - 1))]
    return slices


def load_data_to_list(file_name):
    # user-user user-artist
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            lines = line.replace('\n', '').replace('\r', '').split('\t')
            user = lines[0]
            artist = lines[1]
            data.append((user, artist))
    return data


def distance_matrix_dump(pre_embedding_path, distance_matrix_path):
    with open(pre_embedding_path, 'rb') as file:
        embedding_dict = json.load(file)

    rel_embedding_list = embedding_dict['rel_embeddings']
    ent_embedding_list = embedding_dict['ent_embeddings']
    print('relation_size,entity_size', len(rel_embedding_list), len(ent_embedding_list))

    node_size = len(ent_embedding_list)
    distance_matrix = np.zeros((node_size, node_size))

    for i in range(node_size):
        for j in range(i, node_size):
            x = np.array(ent_embedding_list[i])
            y = np.array(ent_embedding_list[j])
            distance_matrix[i][j] = np.sqrt(np.sum(np.square(x - y)))
            distance_matrix[j][i] = np.sqrt(np.sum(np.square(x - y)))
        print(i)
    np.save(distance_matrix_path, distance_matrix)


def add_all_artists_node_into_graph(g, all_artist_list):
    for pair in all_artist_list:
        user = pair[0]
        artist = pair[1]
        user_node = 'u' + user
        artist_node = 'b' + artist
        g.add_node(user_node)
        g.add_node(artist_node)
        g.add_edge(user_node, artist_node, attr='userbusiness')
    return g


def add_user_artist_interaction_into_graph(g, user_artist_list):
    for pair in user_artist_list:
        user = pair[0]
        artist = pair[1]
        user_node = 'u' + user
        artist_node = 'b' + artist
        g.add_node(user_node)
        g.add_node(artist_node)
        g.add_edge(user_node, artist_node, attr='userbusiness')
    return g


def add_user_friend_into_graph(g, user_friend_list):
    for pair in user_friend_list:
        user_x = pair[0]
        user_y = pair[1]
        user_x_node = 'u' + user_x
        user_y_node = 'u' + user_y
        g.add_node(user_x_node)
        g.add_node(user_y_node)
        g.add_edge(user_x_node, user_y_node, attr='useruser')
    return g


def add_business_city_into_graph(g, business_city_list):
    for pair in business_city_list:
        business = pair[0]
        city = pair[1]
        business_node = 'b' + business
        city_node = 'ci' + city
        g.add_node(business_node)
        g.add_node(city_node)
        g.add_edge(business_node, city_node, attr='businesscity')
    return g


# add movie-movie
def add_business_category_into_graph(g, business_category_list):
    for pair in business_category_list:
        business = pair[0]
        category = pair[1]
        business_node = 'b' + business
        category_node = 'ca' + category
        g.add_node(business_node)
        g.add_node(category_node)
        g.add_edge(business_node, category_node, attr='businesscategory')
    return g


def graph_generation(user_artists_file_name, train_file_name, user_friend_file_name, business_city_file_name,
                     business_category_file_name):

    all_artist_list = load_data_to_list(user_artists_file_name)
    user_artist_list = load_data_to_list(train_file_name)
    user_friend_list = load_data_to_list(user_friend_file_name)
    business_city_list = load_data_to_list(business_city_file_name)
    business_category_list = load_data_to_list(business_category_file_name)
    g = nx.Graph()
    g = add_all_artists_node_into_graph(g, all_artist_list)
    g = add_user_artist_interaction_into_graph(g, user_artist_list)
    g = add_user_friend_into_graph(g, user_friend_list)
    g = add_business_city_into_graph(g, business_city_list)
    g = add_business_category_into_graph(g, business_category_list)
    return g



if __name__ == '__main__':

    user_artists_file_name = 'Yelp/user_business.txt'
    train_file_name = 'Yelp/training.txt'
    user_friend_file_name = 'Yelp/user_friends.txt'
    business_category_file_name = 'Yelp/business_category.txt'
    business_city_file_name = 'Yelp/business_city.txt'

    all_nodes_and_ids_path = 'Yelp/all_nodes_and_ids.pickle'

    # parameters for distance matrix
    pre_embedding_path = 'Yelp/embedding_yelp.vec.json'
    distance_matrix_path = 'Yelp/distance_matrix.npy'

    # generate the graph
    # generate the graph
    g = graph_generation(user_artists_file_name, train_file_name, user_friend_file_name, business_city_file_name,
                         business_category_file_name)
    print('graph finished, total nodes--', len(g.nodes()), 'total edges', len(g.edges()))

    # all nodes into ids
    all_nodes_and_ids = {}
    node_idx = 0
    for node in g.nodes():
        all_nodes_and_ids.update({node: node_idx})
        print(node, node_idx)
        node_idx += 1
    print(len(all_nodes_and_ids))
    with open(all_nodes_and_ids_path,'wb') as file:
         pickle.dump(all_nodes_and_ids,file,0)

    #distance matrix generation
    distance_matrix_dump(pre_embedding_path, distance_matrix_path)


