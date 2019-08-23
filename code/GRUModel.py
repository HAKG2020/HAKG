import torch
import torch.nn as nn

from datetime import datetime


torch.set_default_tensor_type('torch.DoubleTensor')


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """

    def __init__(self, input_dim, hidden_dim):
        super(Propogator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reset_gate = nn.Sequential(
            nn.Linear(self.input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(self.input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(self.input_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, now_input_embedding,dis_pre_embedding):

        a = torch.cat((now_input_embedding, dis_pre_embedding), -1)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((now_input_embedding, r * dis_pre_embedding), 1)
        h_hat = self.tansform(joined_input)
        output = (1 - z) * dis_pre_embedding + z * h_hat
        return output



class GGNN(nn.Module):

    def __init__(self, entity_dim, type_dim, relation_dim, node_size, type_num, relation_num, ent_pre_embedding):
        super(GGNN, self).__init__()
        self.embedding = nn.Embedding(node_size + 1, entity_dim)
        self.embedding.weight = nn.Parameter(ent_pre_embedding)
        self.type_embedding = nn.Embedding(type_num+ 1, type_dim)
        self.rel_type_embedding = nn.Embedding(relation_num + 1, relation_dim)

        self.entity_dim = entity_dim
        self.type_dim = type_dim
        self.relation_dim = relation_dim
        self.input_dim = entity_dim + type_dim
        self.relation_num= relation_num
        self.type_num = type_num
        self.node_size= node_size

        self.in_layer = nn.Sequential(
           nn.Linear(entity_dim + relation_dim, entity_dim),
           nn.ReLU())
        '''
        self.out = nn.Sequential(
            nn.Linear(entity_dim * 2, entity_dim),
            nn.ReLU(),
            nn.Linear(entity_dim, int(entity_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(entity_dim / 2), int(entity_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(entity_dim / 4), 1)
        )
        '''
        self.mlp = nn.Sequential(
            nn.Linear(entity_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.co_attention = nn.Sequential(
            nn.Linear(entity_dim * 2, entity_dim),
            nn.ReLU()
        )

        # Propogation Model
        self.propogator = Propogator(self.input_dim, self.entity_dim)

    #inputs, type_indices, discount_node, rel_matrix, inputs_length, True
    def forward(self, node_slice, type_slice,distance_slice,rel_matrix,input_length, isBatch):

        if isBatch:
            pair_num_a_slice = len(node_slice)

            for pair_id in range(pair_num_a_slice):
                node_num_real = input_length[pair_id]
                distance_one_pair = distance_slice[pair_id][:node_num_real]

                node_num_real = input_length[pair_id]
                entities_one_pair = node_slice[pair_id][:node_num_real]  # 1*path_num*6
                types_one_pair = type_slice[pair_id][:node_num_real]
                rel_matrix_one_pair = rel_matrix[pair_id][:node_num_real,:node_num_real]


                node_embedding = self.embedding(entities_one_pair)
                type_embedding = self.type_embedding(types_one_pair)
                #hidden_states = torch.zeros_like(node_embedding)
                hidden_states= node_embedding.clone().detach()
                for i in range(node_num_real):

                    now_input_embedding = torch.cat((type_embedding[i], node_embedding[i]), -1)
                    rel_vector = rel_matrix_one_pair[i]
                    indices = torch.nonzero(rel_vector).view(1, -1).squeeze(0)

                    rel_embedding = self.rel_type_embedding(rel_vector)

                    pre_rel_embedding = torch.cat((hidden_states, rel_embedding), -1)
                    pre_rel_embedding = torch.index_select(pre_rel_embedding, 0, indices).cuda()
                    pre_embedding = self.in_layer(pre_rel_embedding)

                    #target_entity_embedding = node_embedding[i]
                    #one_tensor = torch.ones((pre_embedding.size()[0], self.entity_dim)).cuda()
                    #stacked_target_entity = one_tensor * target_entity_embedding
                    #distance_vector = torch.norm(stacked_target_entity - pre_embedding, 2, 1).view(1, -1)
                    distance_vector = torch.index_select(distance_one_pair, -1, indices).cuda()
                    distance_vector = distance_vector.double().view(1, -1)
                    #softmax1 = nn.Softmax()

                    #discount_vector = softmax1(-1 * 0.1 * distance_vector)
                    #dis_pre_embedding = torch.mm(discount_vector, pre_embedding)
                    #discount_vector = softmax1(-1 * 0.1 * distance_vector).view(-1, 1)
                    discount_vector = (-1 * 0.1 * distance_vector).view(-1, 1)
                    dis_pre_embedding = torch.max(pre_embedding * discount_vector,0,keepdim=True)[0]
                    print(discount_vector.size())
                    print(dis_pre_embedding.size())
                    updated_embedding = self.propogator(now_input_embedding.view(1, -1), dis_pre_embedding)
                    #node_embedding[i] = updated_embedding
                    hidden_states[i] = updated_embedding


                user_embedding = hidden_states[0]
                item_embedding = hidden_states[-1]
                #subgraph_embedding = hidden_states[-1]
                #subgraph_embedding = torch.mean(hidden_states, dim=0, keepdim=False)

                subgraph_embedding = torch.max(hidden_states, dim=0, keepdim=False)[0]
                user_attention = self.co_attention(torch.cat((user_embedding,subgraph_embedding),-1))
                item_attention = self.co_attention(torch.cat((item_embedding, subgraph_embedding), -1))

                final_user_embedding = user_attention * user_embedding
                final_item_embedding = item_attention * item_embedding
                temp = torch.cat((final_user_embedding,subgraph_embedding,final_item_embedding),-1)
                out = self.mlp(temp).cuda()


                #temp = torch.cat((user_embedding, item_embedding), -1)
                #out = self.out(temp).cuda()
                score_one_pair = torch.sigmoid(out)

                if pair_id == 0:
                    score_slice = score_one_pair.reshape(1)
                else:
                    score_slice = torch.cat([score_slice, score_one_pair.reshape(1)], -1)
            return score_slice

##############################################################################################################3
        else:
            # a.size()[0] a.size()[1]
            if input_length[0] == 0:
                entities_one_pair = node_slice.squeeze()
                node_embedding = self.embedding(entities_one_pair).squeeze()
                user_embedding = node_embedding[0]
                item_embedding = node_embedding[1]
                out = torch.mm(user_embedding.view(1,-1),item_embedding.view(-1,1))
                score_one_pair = torch.sigmoid(out)
                return score_one_pair

            entities_one_pair = node_slice.squeeze()  # 1*path_num*6
            node_num_real = len(entities_one_pair)
            types_one_pair = type_slice.squeeze()
            distance_one_pair = distance_slice

            rel_matrix_one_pair = rel_matrix
            node_embedding = self.embedding(entities_one_pair).squeeze()
            type_embedding = self.type_embedding(types_one_pair)

            hidden_states = node_embedding.clone().detach()
            for i in range(node_num_real):
                now_input_embedding = torch.cat((type_embedding[i], node_embedding[i]), -1)
                rel_vector = rel_matrix_one_pair[i]
                indices = torch.nonzero(rel_vector).view(1, -1).squeeze(0)

                rel_embedding = self.rel_type_embedding(rel_vector)

                pre_rel_embedding = torch.cat((hidden_states, rel_embedding), -1)
                pre_rel_embedding = torch.index_select(pre_rel_embedding, 0, indices).cuda()
                pre_embedding = self.in_layer(pre_rel_embedding)

                # target_entity_embedding = node_embedding[i]
                # one_tensor = torch.ones((pre_embedding.size()[0], self.entity_dim)).cuda()
                # stacked_target_entity = one_tensor * target_entity_embedding
                # distance_vector = torch.norm(stacked_target_entity - pre_embedding, 2, 1).view(1, -1)
                distance_vector = torch.index_select(distance_one_pair, -1, indices).cuda()
                distance_vector = distance_vector.double().view(1, -1)
                #softmax2 = nn.Softmax()

                #discount_vector = softmax2(-1 * 0.1 * distance_vector)
                #dis_pre_embedding = torch.mm(discount_vector, pre_embedding)
                #discount_vector = softmax2(-1 * 0.1 * distance_vector).view(-1, 1)
                discount_vector = (-1 * 0.1 * distance_vector).view(-1, 1)
                dis_pre_embedding = torch.max(pre_embedding * discount_vector, 0, keepdim=True)[0]
                updated_embedding = self.propogator(now_input_embedding.view(1, -1), dis_pre_embedding)
                # node_embedding[i] = updated_embedding
                hidden_states[i] = updated_embedding

            user_embedding = hidden_states[0]
            item_embedding = hidden_states[-1]
            #subgraph_embedding = hidden_states[-1]
            #subgraph_embedding = torch.mean(hidden_states, dim=0, keepdim=False)
            subgraph_embedding = torch.max(hidden_states, dim=0, keepdim=False)[0]
            user_attention = self.co_attention(torch.cat((user_embedding, subgraph_embedding), -1))
            item_attention = self.co_attention(torch.cat((item_embedding, subgraph_embedding), -1))

            final_user_embedding = user_attention * user_embedding
            final_item_embedding = item_attention * item_embedding
            temp = torch.cat((final_user_embedding, subgraph_embedding, final_item_embedding), -1)
            out = self.mlp(temp).cuda()
            score_one_pair = torch.sigmoid(out)


            return score_one_pair




