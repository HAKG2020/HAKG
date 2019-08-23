import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from data_preprocess import load_test_negative_file_as_list
from data_preprocess import getHitRatio
from data_preprocess import getNDCG, generate_batch
from data_preprocess import discountForPathlength
import heapq
import numpy

torch.set_default_tensor_type('torch.DoubleTensor')
from datetime import datetime

class IGRUTrain(object):

    def __init__(self, model, iteration, learning_rate, weight_decay,
                 all_label_list, all_nodes_and_ids, positive_dict,
                 negative_dict, test_negative_path_dict,type_num,relation_num,alpha):
        super(IGRUTrain, self).__init__()

        self.model = model
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.batch_size = 256
        self.alpha = alpha
        self.weight_decay = weight_decay

        self.positive_dict = positive_dict
        self.negative_dict = negative_dict
        self.test_negative_path_dict = test_negative_path_dict

        self.positive_label = list(self.positive_dict.keys())
        self.negative_label = list(self.negative_dict.keys())
        self.all_label_list = all_label_list

        self.all_nodes_and_ids = all_nodes_and_ids
        self.type_num = type_num
        self.relation_num= relation_num

    def get_relation_id(self, x,y):

        if 'u' in x and 'u' in y:
            return 1

        if 'i' in x and 'i' in y:
            return 2

        if ('u' in x and 'i' in y) or ('i' in x and 'u' in y):
            return 3

        if ('i' in x and 'g' in y) or ('g' in x and 'i' in y):
            return 4

    def get_type_id(self, x):
        if 'u' in x:
            return 0
        if 'i' in x:
            return 1
        if 'g' in x:
            return 2



    def prepareDataForOnePair(self, pair, subgraphs_map, all_nodes_and_ids, alpha):


        key1 = pair[0] + '-' + pair[1]

        sequence = subgraphs_map[key1]
        nodes = sequence[1][0]
        maxlen = len(nodes)
        distance_list = sequence[2][0]
        distance_vector = numpy.zeros((1, maxlen))
        distance_vector[0,:] = numpy.array(distance_list)
        node_id_vector = numpy.zeros((1, maxlen), dtype='int64')
        type_id_vector = numpy.zeros((1, maxlen), dtype='int64')
        relation_matrix = numpy.zeros((maxlen, maxlen))

        dependency_matrix = numpy.zeros((maxlen, maxlen))  # @UndefinedVariable
        depend = sequence[0][0]
        nodes = sequence[1][0]
        idx = 0
        node_dict = {}
        for node in nodes:
            node_dict.update({node: idx})
            node_id_vector[0,idx] = all_nodes_and_ids[node]
            type_id_vector[0, idx] = self.get_type_id(node)
            idx += 1
        for dep in depend:
            dependency_matrix[node_dict[dep[1]]][node_dict[dep[0]]] = 1
            relation_matrix[node_dict[dep[1]]][node_dict[dep[0]]] = self.get_relation_id(dep[1],dep[0])
        for j in range(maxlen):
            if dependency_matrix[j].sum() == 0:
                dependency_matrix[j][j] = 1
                relation_matrix[j][j] = 1
        return node_id_vector, type_id_vector, distance_vector,relation_matrix, maxlen


    def train_relation(self):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        testRatings, testNegatives = load_test_negative_file_as_list('../ml/test_negative.txt')
        #########################################train#################################################
        node_size = len(self.all_nodes_and_ids)
        length = len(self.all_label_list)

        # set batch_size
        slices = generate_batch(length, self.batch_size)

        for epoch in range(self.iteration):
            start = datetime.now()
            print(epoch)
            running_loss = 0.0
            num = 0

            for slice in slices:
                print('hhhh')
                node_id_slice = []
                type_id_slice = []
                discount_slice =[]
                relation_matrix_slice = []
                label_slice = []
                inputs_length = []

                t1 = datetime.now()
                for j in slice:
                    ui_pair = self.all_label_list[j]
                    user_name = ui_pair[0]
                    item_name = ui_pair[1]
                    pair = (user_name, item_name)
                    key = user_name +'-'+item_name
                    if key in self.positive_label:
                        node_id_list, type_id_list, discount_list,relation_matrix, node_num_a_sequence = \
                            self.prepareDataForOnePair(pair, self.positive_dict, self.all_nodes_and_ids, self.alpha)
                        label_one = np.array([1])

                    else:
                        node_id_list, type_id_list, discount_list,relation_matrix,node_num_a_sequence = \
                            self.prepareDataForOnePair(pair, self.negative_dict, self.all_nodes_and_ids, self.alpha)
                        label_one = np.array([0])


                    if node_id_list is None:
                        continue
                    discount_slice.append(discount_list)
                    node_id_slice.append(node_id_list)  # need pad
                    type_id_slice.append(type_id_list)  # need pad
                    inputs_length.append(node_num_a_sequence)

                    label_slice.append(label_one)
                    relation_matrix_slice.append(relation_matrix)

                # pad inputs
                padding_index_node = node_size
                padding_index_type = self.type_num
                max_len = max(inputs_length)
                batch_size = len(node_id_slice)

                padded_inputs = np.ones((batch_size, max_len)) * padding_index_node
                padded_type_indices = np.ones((batch_size, max_len)) * padding_index_type
                padded_discount = np.zeros((batch_size, max_len))
                padded_relation_matrix = []

                for i, x_len in enumerate(inputs_length):
                    sequence = node_id_slice[i]
                    type = type_id_slice[i]
                    discount = discount_slice[i]
                    padded_inputs[i, 0:x_len] = sequence[0,:x_len]
                    padded_type_indices[i, 0:x_len] = type[0,:x_len]
                    padded_discount[i, 0:x_len] = discount[0,:x_len]
                    padded_relation_matrix_one = np.zeros((max_len, max_len))
                    Real_relation_matrix_one = relation_matrix_slice[i]
                    padded_relation_matrix_one[0:x_len, 0:x_len] = Real_relation_matrix_one[0:x_len, 0:x_len]
                    padded_relation_matrix.append(torch.LongTensor(padded_relation_matrix_one))

                inputs = Variable(torch.LongTensor(padded_inputs)).cuda()
                type_indices = Variable(torch.LongTensor(padded_type_indices)).cuda()
                discount_node = Variable(torch.DoubleTensor(padded_discount)).cuda()
                label = Variable(torch.LongTensor(label_slice)).cuda()
                inputs_length = torch.LongTensor(inputs_length).cuda()
                rel_matrix = Variable(torch.stack(padded_relation_matrix)).cuda()

                out = self.model(inputs, type_indices, discount_node, rel_matrix, inputs_length, True)
                out = out.squeeze()
                loss = criterion(out, label.squeeze().double())

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num += 1
                print('ml_mmtrue', num, 'loss: ', loss.item(), 'running loss', running_loss)
            

            end = datetime.now()
            duration = str((end - start).seconds)
            print('epoch[' + str(epoch) + ']: loss is ' + str(running_loss))

            # save model
            path = 'ml_dag/model_epoch_' + str(epoch) + '.pt'
            torch.save(self.model,path)

            verbose = 1
            if epoch % verbose == 0:
                (hits_1, ndcgs_1), (hits_2, ndcgs_2), (hits_3, ndcgs_3), (hits_4, ndcgs_4), (hits_5, ndcgs_5), (
                    hits_6, ndcgs_6), (hits_7, ndcgs_7), (hits_8, ndcgs_8), (hits_9, ndcgs_9), (hits_10, ndcgs_10), \
                (hits_11, ndcgs_11), (hits_12, ndcgs_12), (hits_13, ndcgs_13), (hits_14, ndcgs_14), (
                    hits_15, ndcgs_15) = self.evaluate_model(self.model, testRatings, testNegatives)
                hr_1, ndcg_1 = np.array(hits_1).mean(), np.array(ndcgs_1).mean()
                hr_2, ndcg_2 = np.array(hits_2).mean(), np.array(ndcgs_2).mean()
                hr_3, ndcg_3 = np.array(hits_3).mean(), np.array(ndcgs_3).mean()
                hr_4, ndcg_4 = np.array(hits_4).mean(), np.array(ndcgs_4).mean()

                hr_5, ndcg_5 = np.array(hits_5).mean(), np.array(ndcgs_5).mean()
                hr_6, ndcg_6 = np.array(hits_6).mean(), np.array(ndcgs_6).mean()
                hr_7, ndcg_7 = np.array(hits_7).mean(), np.array(ndcgs_7).mean()
                hr_8, ndcg_8 = np.array(hits_8).mean(), np.array(ndcgs_8).mean()
                hr_9, ndcg_9 = np.array(hits_9).mean(), np.array(ndcgs_9).mean()
                hr_10, ndcg_10 = np.array(hits_10).mean(), np.array(ndcgs_10).mean()
                hr_11, ndcg_11 = np.array(hits_11).mean(), np.array(ndcgs_11).mean()
                hr_12, ndcg_12 = np.array(hits_12).mean(), np.array(ndcgs_12).mean()
                hr_13, ndcg_13 = np.array(hits_13).mean(), np.array(ndcgs_13).mean()
                hr_14, ndcg_14 = np.array(hits_14).mean(), np.array(ndcgs_14).mean()
                hr_15, ndcg_15 = np.array(hits_15).mean(), np.array(ndcgs_15).mean()
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_1, ndcg_1))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_2, ndcg_2))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_3, ndcg_3))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_4, ndcg_4))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_5, ndcg_5))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_6, ndcg_6))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_7, ndcg_7))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_8, ndcg_8))
                print('Iteration %d : HR_5 = %.4f, NDCG_5 = %.4f '
                      % (epoch, hr_9, ndcg_9))
                print('Iteration %d : HR_10 = %.4f, NDCG_10 = %.4f '
                      % (epoch, hr_10, ndcg_10))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_11, ndcg_11))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_12, ndcg_12))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_13, ndcg_13))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_14, ndcg_14))
                print('Iteration %d : HR_15 = %.4f, NDCG_15 = %.4f '
                      % (epoch, hr_15, ndcg_15))

                line_1 = 'epoch:  ' + str(epoch) + '  hr_1: ' + str(hr_1) + '  ndcg_1: ' + str(ndcg_1) + '\n'
                line_2 = 'epoch:  ' + str(epoch) + '  hr_2: ' + str(hr_2) + '  ndcg_2: ' + str(ndcg_2) + '\n'
                line_3 = 'epoch:  ' + str(epoch) + '  hr_3: ' + str(hr_3) + '  ndcg_3: ' + str(ndcg_3) + '\n'
                line_4 = 'epoch:  ' + str(epoch) + '  hr_4: ' + str(hr_4) + '  ndcg_4: ' + str(ndcg_4) + '\n'

                line_5 = 'epoch:  ' + str(epoch) + '  hr_5: ' + str(hr_5) + '  ndcg_5: ' + str(ndcg_5) + '\n'
                line_6 = 'epoch:  ' + str(epoch) + '  hr_6: ' + str(hr_6) + '  ndcg_6: ' + str(ndcg_6) + '\n'
                line_7 = 'epoch:  ' + str(epoch) + '  hr_7: ' + str(hr_7) + '  ndcg_7: ' + str(ndcg_7) + '\n'
                line_8 = 'epoch:  ' + str(epoch) + '  hr_8: ' + str(hr_8) + '  ndcg_8: ' + str(ndcg_8) + '\n'
                line_9 = 'epoch:  ' + str(epoch) + '  hr_9: ' + str(hr_9) + '  ndcg_9: ' + str(ndcg_9) + '\n'
                line_10 = 'epoch:  ' + str(epoch) + '  hr_10: ' + str(hr_10) + '  ndcg_10: ' + str(ndcg_10) + '\n'

                line_11 = 'epoch:  ' + str(epoch) + '  hr_11: ' + str(hr_11) + '  ndcg_11: ' + str(ndcg_11) + '\n'
                line_12 = 'epoch:  ' + str(epoch) + '  hr_12: ' + str(hr_12) + '  ndcg_12: ' + str(ndcg_12) + '\n'
                line_13 = 'epoch:  ' + str(epoch) + '  hr_13: ' + str(hr_13) + '  ndcg_13: ' + str(ndcg_13) + '\n'
                line_14 = 'epoch:  ' + str(epoch) + '  hr_14: ' + str(hr_14) + '  ndcg_14: ' + str(ndcg_14) + '\n'
                line_15 = 'epoch:  ' + str(epoch) + '  hr_15: ' + str(hr_15) + '  ndcg_15: ' + str(ndcg_15) + '\n'
                loss_str = 'epoch:  ' + str(epoch) + 'loss is ' + str(running_loss) + '\n' + '\n'
                results_path = 'ml_dag/results_epoch_' + str(epoch) + '.txt'

                with open(results_path, 'w') as file:
                    file.write(line_1)
                    file.write(line_2)
                    file.write(line_3)
                    file.write(line_4)
                    file.write(line_5)
                    file.write(line_6)
                    file.write(line_7)
                    file.write(line_8)
                    file.write(line_9)

                    file.write(line_10)
                    file.write(line_11)
                    file.write(line_12)
                    file.write(line_13)
                    file.write(line_14)
                    file.write(line_15)
                    file.write(loss_str)


    def evaluate_model(self, model, testRatings, testNegatives):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """

        global _model
        global _testRatings
        global _testNegatives

        _testRatings = testRatings
        _testNegatives = testNegatives

        hits_1, hits_2, hits_3, hits_4, hits_5, hits_6, hits_7, hits_8, hits_9, hits_10, hits_11, hits_12, hits_13, hits_14, hits_15 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        ndcgs_1, ndcgs_2, ndcgs_3, ndcgs_4, ndcgs_5, ndcgs_6, ndcgs_7, ndcgs_8, ndcgs_9, ndcgs_10, ndcgs_11, ndcgs_12, ndcgs_13, ndcgs_14, ndcgs_15 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        # Single thread

        for idx in range(len(_testRatings)):
            print('test num', idx)

            (hr_1, ndcg_1), (hr_2, ndcg_2), (hr_3, ndcg_3), (hr_4, ndcg_4), (hr_5, ndcg_5), (hr_6, ndcg_6), \
            (hr_7, ndcg_7), (hr_8, ndcg_8), (hr_9, ndcg_9), (hr_10, ndcg_10), (hr_11, ndcg_11), (hr_12, ndcg_12), \
            (hr_13, ndcg_13), (hr_14, ndcg_14), (hr_15, ndcg_15) = self.eval_one_rating(idx)
            hits_1.append(hr_1)
            ndcgs_1.append(ndcg_1)
            hits_2.append(hr_2)
            ndcgs_2.append(ndcg_2)
            hits_3.append(hr_3)
            ndcgs_3.append(ndcg_3)
            hits_4.append(hr_4)
            ndcgs_4.append(ndcg_4)
            hits_5.append(hr_5)
            ndcgs_5.append(ndcg_5)
            hits_6.append(hr_6)
            ndcgs_6.append(ndcg_6)
            hits_7.append(hr_7)
            ndcgs_7.append(ndcg_7)
            hits_8.append(hr_8)
            ndcgs_8.append(ndcg_8)
            hits_9.append(hr_9)
            ndcgs_9.append(ndcg_9)
            hits_10.append(hr_10)
            ndcgs_10.append(ndcg_10)
            hits_11.append(hr_11)
            ndcgs_11.append(ndcg_11)
            hits_12.append(hr_12)
            ndcgs_12.append(ndcg_12)
            hits_13.append(hr_13)
            ndcgs_13.append(ndcg_13)
            hits_14.append(hr_14)
            ndcgs_14.append(ndcg_14)
            hits_15.append(hr_15)
            ndcgs_15.append(ndcg_15)

        return (hits_1, ndcgs_1), (hits_2, ndcgs_2), (hits_3, ndcgs_3), (hits_4, ndcgs_4), (hits_5, ndcgs_5), \
               (hits_6, ndcgs_6), (hits_7, ndcgs_7), (hits_8, ndcgs_8), (hits_9, ndcgs_9), (hits_10, ndcgs_10), \
               (hits_11, ndcgs_11), (hits_12, ndcgs_12), (hits_13, ndcgs_13), (hits_14, ndcgs_14), (hits_15, ndcgs_15)

    def eval_one_rating(self, idx):
        # di idx interaction in test

        rating = _testRatings[idx]
        items = _testNegatives[idx]
        user = rating[0]
        gtItem = rating[1]
        items.append(gtItem)

        # Get prediction scores
        map_item_score = {}
        predictions = []
        for item in items:
            score_for_one_pair = self.Compute_score_for_one_pair(user, item)
            predictions.append(score_for_one_pair)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        items.pop()

        # Evaluate top rank list
        ranklist_1 = heapq.nlargest(1, map_item_score, key=map_item_score.get)
        ranklist_2 = heapq.nlargest(2, map_item_score, key=map_item_score.get)
        ranklist_3 = heapq.nlargest(3, map_item_score, key=map_item_score.get)
        ranklist_4 = heapq.nlargest(4, map_item_score, key=map_item_score.get)
        ranklist_5 = heapq.nlargest(5, map_item_score, key=map_item_score.get)
        ranklist_6 = heapq.nlargest(6, map_item_score, key=map_item_score.get)
        ranklist_7 = heapq.nlargest(7, map_item_score, key=map_item_score.get)
        ranklist_8 = heapq.nlargest(8, map_item_score, key=map_item_score.get)
        ranklist_9 = heapq.nlargest(9, map_item_score, key=map_item_score.get)
        ranklist_10 = heapq.nlargest(10, map_item_score, key=map_item_score.get)
        ranklist_11 = heapq.nlargest(11, map_item_score, key=map_item_score.get)
        ranklist_12 = heapq.nlargest(12, map_item_score, key=map_item_score.get)
        ranklist_13 = heapq.nlargest(13, map_item_score, key=map_item_score.get)
        ranklist_14 = heapq.nlargest(14, map_item_score, key=map_item_score.get)
        ranklist_15 = heapq.nlargest(15, map_item_score, key=map_item_score.get)
        hr_1 = getHitRatio(ranklist_1, gtItem)
        ndcg_1 = getNDCG(ranklist_1, gtItem)
        hr_2 = getHitRatio(ranklist_2, gtItem)
        ndcg_2 = getNDCG(ranklist_2, gtItem)
        hr_3 = getHitRatio(ranklist_3, gtItem)
        ndcg_3 = getNDCG(ranklist_3, gtItem)
        hr_4 = getHitRatio(ranklist_4, gtItem)
        ndcg_4 = getNDCG(ranklist_4, gtItem)
        hr_5 = getHitRatio(ranklist_5, gtItem)
        ndcg_5 = getNDCG(ranklist_5, gtItem)
        hr_6 = getHitRatio(ranklist_6, gtItem)
        ndcg_6 = getNDCG(ranklist_6, gtItem)
        hr_7 = getHitRatio(ranklist_7, gtItem)
        ndcg_7 = getNDCG(ranklist_7, gtItem)
        hr_8 = getHitRatio(ranklist_8, gtItem)
        ndcg_8 = getNDCG(ranklist_8, gtItem)
        hr_9 = getHitRatio(ranklist_9, gtItem)
        ndcg_9 = getNDCG(ranklist_9, gtItem)
        hr_10 = getHitRatio(ranklist_10, gtItem)
        ndcg_10 = getNDCG(ranklist_10, gtItem)
        hr_11 = getHitRatio(ranklist_11, gtItem)
        ndcg_11 = getNDCG(ranklist_11, gtItem)
        hr_12 = getHitRatio(ranklist_12, gtItem)
        ndcg_12 = getNDCG(ranklist_12, gtItem)
        hr_13 = getHitRatio(ranklist_13, gtItem)
        ndcg_13 = getNDCG(ranklist_13, gtItem)
        hr_14 = getHitRatio(ranklist_14, gtItem)
        ndcg_14 = getNDCG(ranklist_14, gtItem)
        hr_15 = getHitRatio(ranklist_15, gtItem)
        ndcg_15 = getNDCG(ranklist_15, gtItem)
        return (hr_1, ndcg_1), (hr_2, ndcg_2), (hr_3, ndcg_3), (hr_4, ndcg_4), (hr_5, ndcg_5), \
               (hr_6, ndcg_6), (hr_7, ndcg_7), (hr_8, ndcg_8), (hr_9, ndcg_9), (hr_10, ndcg_10), (hr_11, ndcg_11), \
               (hr_12, ndcg_12), (hr_13, ndcg_13), (hr_14, ndcg_14), (hr_15, ndcg_15)

    def Compute_score_for_one_pair(self, user, item):
        user_name = 'u' + user
        item_name = 'i' + item

        pair = (user_name, item_name)


        node_id_list, type_id_list,  discount_list, relation_matrix, node_num_a_sequence= \
            self.prepareDataForOnePair(pair, self.test_negative_path_dict, self.all_nodes_and_ids, self.alpha)

        if node_id_list is None:
            user_id = self.all_nodes_and_ids[user_name]
            item_id = self.all_nodes_and_ids[item_name]
            node_num_a_sequence = 0
            node_id_list = numpy.array([user_id,item_id])
            type_id_list = numpy.array([0,1])
            discount_list =  numpy.zeros((1, 2))
            relation_matrix = numpy.zeros((2, 2))

        inputs_length = [node_num_a_sequence]
        inputs_length = torch.LongTensor(inputs_length).cuda()
        inputs = torch.LongTensor(node_id_list).cuda()
        type_indices = torch.LongTensor(type_id_list).cuda()
        discount_node = torch.DoubleTensor(discount_list).cuda()
        rel_matrix = torch.LongTensor(relation_matrix).cuda()


        score = self.model(inputs, type_indices,discount_node, rel_matrix, inputs_length, False)
        score = score.data.cpu().numpy()
        score = score[0]

        return score
