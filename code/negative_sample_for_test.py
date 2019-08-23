# This is used to sample negative movies that user has not interactions with, so as to balance model training process

import argparse
import math
import numpy as np
from random import randint
import pickle


def load_train_data(file):
    train_dict = {}
    all_artist_list = []
    for line in file:
        lines = line.replace('\n', '').replace('\r', '').split('\t')
        user = lines[0]
        artist = lines[1]
        if user not in train_dict:
            train_dict.update({user: [artist]})
        else:
            train_dict[user].append(artist)
        if artist not in all_artist_list:
            all_artist_list.append(artist)
    return train_dict, all_artist_list


def test_negative_sample(all_dict,all_artist_list,test_dict, all_test_artist_list, N, test_negative_sample_file):

    all_artist_num = len(all_artist_list)
    print('all_artist the whole ', all_artist_num)
    i = 0
    for user in test_dict:
        user_node = 'u' + user
        print(user_node, i)
        i += 1
        test_list_this_user = test_dict[user]
        positive_list_this_user = all_dict[user]
        for test_item in test_list_this_user:
            test_negative_list = []
            flag = 0
            while (len(test_negative_list) < N):
                if(flag > 100):
                    break
                else:
                    flag += 1
                    negative_index = np.random.randint(0, (all_artist_num - 1))
                    negative_artist = str(all_artist_list[negative_index])
                    if negative_artist not in positive_list_this_user and negative_artist not in test_negative_list:
                        test_negative_list.append(negative_artist)

            string = ''
            for negative_item in test_negative_list:
                string += '\t'
                string += negative_item

            line = user + '\t' + test_item + string + '\n'
            test_negative_sample_file.write(line)

    test_negative_sample_file.close()



if __name__ == '__main__':

    user_artists_file_name = 'ml/user_movies.txt'
    train_file_name = 'ml/training.txt'
    test_file_name = 'ml/test.txt'

    all_nodes_and_ids_path = 'ml/all_nodes_and_ids.pickle'
    N = 50
    test_negative_sample_file_name = 'ml/test_negative.txt'

    all_dict, all_artist_list = load_train_data((open(user_artists_file_name, 'r')))
    test_dict, all_test_artist_list = load_train_data((open(test_file_name, 'r')))
    train_dict, all_train_artist_list = load_train_data((open(train_file_name, 'r')))
    test_negative_sample(all_dict,all_artist_list,test_dict, all_test_artist_list, N, open(test_negative_sample_file_name, 'w'))

