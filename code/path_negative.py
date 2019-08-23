import networkx as nx
import pickle
import json
import random
from data_preprocess import *
import gc

import heapq

def get_distance(node_x_idx,node_y_idx):
    global ent_embedding_list
    x = np.array(ent_embedding_list[node_x_idx])
    y = np.array(ent_embedding_list[node_y_idx])
    return np.sum(np.square(x - y))

def path_generation_for_one_pair(g, all_nodes_and_ids, distance_matrix, user_node,artist_node, path_len_constrain,
                                 sample_path_num, save_path_file):
    global negative_dict
    # global node_size_list
    pathidx_and_distance_dict = {}
    path_list = list(nx.all_simple_paths(g, user_node, artist_node, cutoff=path_len_constrain))

    if len(path_list) != 0:
        print(user_node, artist_node, len(path_list))

        if user_node not in negative_dict:
            negative_dict.update({user_node:[artist_node]})
        else:
            negative_dict[user_node].append(artist_node)
        path_idx = 0
        for path in path_list:
            sum = 0
            node_count = len(path)  # u1,i1,g1,i2 -- 4
            for i in range(node_count - 1):  # 0,1,2,
                node_x = path[i]
                node_y = path[i + 1]
                node_x_idx = int(all_nodes_and_ids[node_x])
                node_y_idx = int(all_nodes_and_ids[node_y])
                distance = get_distance(node_x_idx, node_y_idx)
                sum += distance
            pathidx_and_distance_dict.update({path_idx: sum})
            path_idx += 1
        # get topK path idx
        if len(path_list) > sample_path_num:
            ranklist_idx = heapq.nsmallest(sample_path_num, pathidx_and_distance_dict, key=pathidx_and_distance_dict.get)
        else:
            ranklist_idx = heapq.nsmallest(len(path_list), pathidx_and_distance_dict, key=pathidx_and_distance_dict.get)
        # get topK path from path_list using path_idx
        #score = []
        for path_idx in ranklist_idx:
            path = path_list[path_idx]
            #score.append(pathidx_and_distance_dict[path_idx])
            line = ",".join(path) + '\n'
            save_path_file.write(line)
        #print(score)
    else:
        print('None')
        line = user_node + ',' + artist_node+','+'None'+'\n'
        save_path_file.write(line)

def path_generation(g, rating, all_nodes_and_ids, distance_matrix, path_len_constrain, sample_path_num,
                    save_path_file_name):
    pair_idx = 0
    save_path_file = open(save_path_file_name, 'w')
    for pair in rating:
        user_id = pair[0]
        artist_id = pair[1]
        user_node = 'u' + user_id
        artist_node = 'i' + artist_id

        if g.has_node(user_node) and g.has_node(artist_node):
            path_generation_for_one_pair(g, all_nodes_and_ids, distance_matrix, user_node, artist_node,
                                         path_len_constrain, sample_path_num, save_path_file)
            # print number
            pair_idx += 1
            print(pair_idx, pair)

    save_path_file.close()


def load_data(file):
    data = []
    for line in file:
        lines = line.replace('\n', '').replace('\r', '').split('\t')
        user = lines[0]
        movie = lines[1]
        data.append((user, movie))
    return data


if __name__ == '__main__':

    negative_file_name = 'ml/negative.txt'
    user_artists_file_name = 'ml/user_movies.txt'
    train_file_name = 'ml/training.txt'
    user_friend_file_name = 'ml/user_friends.txt'
    artist_tag_file_name = 'ml/movie_genre.txt'
    item_item_file_name = 'ml/movie_movie.txt'
    pre_embedding_path = 'ml/embedding_ml1m_64.vec.json'
    all_nodes_and_ids_path = 'ml/all_nodes_and_ids.pickle'
    item_side_information_file_name = 'ml/movie_features.txt'
    # parameters for distance matrix

    distance_matrix_path = 'ml/distance_matrix.npy'

    positive_path_file_name = 'ml/positive_path.txt'
    negative_path_file_name = 'ml/negative_path.txt'
    test_negative_path_file_name = 'ml/test_negative_path.txt'


    path_len_constrain = 3
    sample_path_num = 30


    # generate the graph
    g = graph_generation(user_artists_file_name, train_file_name, item_side_information_file_name)
    print('graph finished, total nodes--', len(g.nodes()), 'total edges', len(g.edges()))


    with open(all_nodes_and_ids_path, 'rb') as file:
        all_nodes_and_ids = pickle.load(file)
    print('load all nodes and ids finished')



    distance_matrix = 0
    # pre_train_embedding
    with open(pre_embedding_path, 'rb') as file:
        embedding_dict = json.load(file)
    rel_embedding_list = embedding_dict['rel_embeddings']
    ent_embedding_list = embedding_dict['ent_embeddings']
    print('relation_size,entity_size', len(rel_embedding_list), len(ent_embedding_list))

    negative_rating = load_data(open(negative_file_name, 'r'))
    negative_dict = {}
    path_generation(g, negative_rating, all_nodes_and_ids, distance_matrix, path_len_constrain, sample_path_num,
                    negative_path_file_name)


    print('finished negative')











