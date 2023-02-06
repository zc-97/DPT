import torch
import random
import torch.nn as nn
from torch.nn import Module, Parameter, init
import torch.nn.functional as F
import numpy as np
import math
import time
from torch.autograd import Variable
from Params import args


class myModel1(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats, behavior_matsT, i_in, i_out, u_in):
        super(myModel1, self).__init__()

        self.userNum = userNum
        self.itemNum = itemNum


        # behavior_num = behavior - 1

        # if args.prompt and not args.denoise_tune:
        self.prompt = torch.nn.Embedding(1, args.dim)
        nn.init.xavier_uniform_(self.prompt.weight)
        self.behavior = behavior
        self.behavior_mats = behavior_mats
        self.behavior_matsT = behavior_matsT
        self.i_in = i_in
        self.i_out = i_out
        self.u_in = u_in

        self.sigmoid = nn.Sigmoid()


        self.user_embedding, self.item_embedding, self.beh_embedding = self.init_embedding()


        self.gcn = GCNv5(self.userNum, self.itemNum, self.behavior, self.behavior_mats, self.behavior_matsT,
                         self.i_in, self.i_out, self.u_in,
                         self.user_embedding.weight, self.item_embedding.weight, self.beh_embedding.weight, self.prompt.weight)

        self.denoising = graphDenoising(args.dim,self.behavior)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, args.opt_base_lr, args.opt_max_lr,
                                                           step_size_up=5, step_size_down=10, mode='triangular',
                                                           gamma=0.99, scale_fn=None, scale_mode='cycle',
                                                           cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
                                                           last_epoch=-1)

    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, args.dim)
        item_embedding = torch.nn.Embedding(self.itemNum, args.dim)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)
        # if not args.prompt:
        #     beh_embedding = torch.nn.Embedding(self.behavior, args.dim)
        #
        # else:
        #     beh_embedding = torch.nn.Embedding(1,args.dim)

        beh_embedding = torch.nn.Embedding(self.behavior - 1, args.dim)



        nn.init.xavier_uniform_(beh_embedding.weight)

        return user_embedding, item_embedding, beh_embedding
    if not args.prompt:
        def forward(self, user=0,pos=0,neg=0, is_denoise=False):


            user_embed, item_embed,user_embeds, item_embeds = self.gcn()

            if is_denoise:

                sub_user_emb_list = [None] * self.behavior
                sub_pos_emb_list = [None] * self.behavior

                sub_neg_emb_list = [None] * self.behavior
                for beh in range(self.behavior):
                    sub_user_emb_list[beh] = user_embeds[beh][user]
                    sub_pos_emb_list[beh] = item_embeds[beh][pos[beh]]
                    sub_neg_emb_list[beh] = item_embeds[beh][neg[beh]]
                graphs, graphsT = self.denoising(sub_user_emb_list,sub_pos_emb_list,sub_neg_emb_list, self.beh_embedding.weight, self.prompt.weight)
            else:
                graphs = 0

            return user_embed, item_embed, user_embeds, item_embeds, graphs

            # return enhanced_user_emb, enhanced_item_emb, user_embeds, item_embeds, graphs
    else:
        def forward(self, user_set=0,item_list=0, is_denoise=False):
            user_embed, item_embed, user_embeds, item_embeds = self.gcn()
            return user_embed, item_embed, user_embeds, item_embeds, 0

class GCNv5(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats, behavior_matsT, i_in, i_out, u_in, u_emb, i_emb, beh_emb, prompt_emb):
        super(GCNv5, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.dim
        self.behavior = range(behavior)
        self.behavior_mats = behavior_mats
        self.behavior_matsT = behavior_matsT

        self.i_in = i_in
        self.i_out = i_out
        self.u_in = u_in

        # self.user_embedding, self.item_embedding = self.init_embedding()
        self.user_embedding = u_emb
        self.item_embedding = i_emb
        self.behavior_embeddings = beh_emb
        self.prompt_embedding = prompt_emb

        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()

        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)

        gnn_layer = args.gnn_layer

        self.gnn_layer = eval(gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.gnn_layer)):
            self.layers.append(GCNLayerv5(args.dim, args.dim, self.userNum, self.itemNum, len(self.behavior),
                                        self.behavior_mats, self.behavior_matsT, self.i_in, self.i_out, self.u_in))


    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, args.dim)
        item_embedding = torch.nn.Embedding(self.itemNum, args.dim)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))
        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer)) * args.dim, args.dim))
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer)) * args.dim, args.dim))
        i_input_w = nn.Parameter(torch.Tensor(args.dim, args.dim))
        u_input_w = nn.Parameter(torch.Tensor(args.dim, args.dim))

        init.xavier_uniform_(i_concatenation_w)
        init.xavier_uniform_(u_concatenation_w)
        init.xavier_uniform_(i_input_w)
        init.xavier_uniform_(u_input_w)


        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w
    def forward(self, user_embedding_input=None, item_embedding_input=None):
        all_user_embeddings = []
        all_item_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []


        user_embedding = self.user_embedding
        item_embedding = self.item_embedding
        beh_embedding_list = self.behavior_embeddings
        #
        # if args.prompt and not args.denoise_tune:
        #     user_embedding = user_embedding + self.prompt_embedding
        #     item_embedding = item_embedding + self.prompt_embedding

        for i, layer in enumerate(self.layers):
            # user_emb, item_emb, user_embs, item_embs = layer(user_embedding, item_embedding)
            user_embedding,item_embedding, user_embeddings,item_embeddings = layer(user_embedding, item_embedding, beh_embedding_list, self.prompt_embedding)
            # if args.prompt and args.deep and not args.denoise_tune:
            #     user_embedding = user_embedding + beh_embedding_list[-1]


            norm_user_embeddings = F.normalize(user_embedding, p=2, dim=1)
            norm_item_embeddings = F.normalize(item_embedding, p=2, dim=1)

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)


            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)


        user_embedding = torch.cat(all_user_embeddings, dim=1)
        item_embedding = torch.cat(all_item_embeddings, dim=1)


        user_embeddings = torch.cat(all_user_embeddingss, dim=2)
        item_embeddings = torch.cat(all_item_embeddingss, dim=2)

        user_embedding = torch.matmul(user_embedding, self.u_concatenation_w)
        item_embedding = torch.matmul(item_embedding, self.i_concatenation_w)
        user_embeddings = torch.matmul(user_embeddings, self.u_concatenation_w)
        item_embeddings = torch.matmul(item_embeddings, self.i_concatenation_w)


        return user_embedding, item_embedding, user_embeddings, item_embeddings  # [31882, 16], [31882, 16], [4, 31882, 16], [4, 31882, 16]

class GCNLayerv5(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behaviorNum, behavior_mats, behavior_mats_t, i_in, i_out,
                 u_in):
        super(GCNLayerv5, self).__init__()

        self.behaviorNum = behaviorNum
        self.behavior_mats = behavior_mats
        self.behavior_mats_t = behavior_mats_t

        self.userNum = userNum
        self.itemNum = itemNum

        self.i_in = i_in
        self.i_out = i_out
        self.u_in = u_in

        self.act = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(2 * in_dim, out_dim))
        self.uu_w = nn.Parameter(torch.Tensor(2 * in_dim, out_dim))

        self.W1 = nn.Parameter(torch.Tensor(args.dim, args.dim))
        self.W2 = nn.Parameter(torch.Tensor(args.dim, args.dim))

        self.conv_layer = nn.Conv2d(1, 1, (1, 2), bias=True)
        self.conv_layer_user = nn.Conv2d(1, 1, (1, 2), bias=True)
        self.feat_drop = nn.Dropout(args.drop_rate)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU(0.2)

        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)
        init.xavier_uniform_(self.ii_w)
        init.xavier_uniform_(self.uu_w)
        init.xavier_uniform_(self.conv_layer.weight)
        init.xavier_uniform_(self.conv_layer_user.weight)
        init.xavier_uniform_(self.W1)
        init.xavier_uniform_(self.W2)


    def forward(self, user_embedding, item_embedding, beh_embeddings, prompt_emb):

        user_embedding_list = [None] * self.behaviorNum
        item_embedding_list = [None] * self.behaviorNum

        for i in range(self.behaviorNum):
            if i==self.behaviorNum-1:
                prompt = prompt_emb
            else:
                prompt = 0.0


            if args.prompt:
                if not args.denoise_tune:

                    # 目前IJCAI性能最好的
                    # if i == self.behaviorNum - 1:
                    #     # pass
                    #     #
                    #     item_embedding = (item_embedding @ prompt.t()) / (prompt @ prompt.t()) * prompt
                    #     user_embedding = (user_embedding @ prompt.t()) / (prompt @ prompt.t()) * prompt
                    #
                    # else:
                    #     pass
                    # user_embedding_list[i] = torch.spmm(self.behavior_mats[i], item_embedding)
                    # item_embedding_list[i] = torch.spmm(self.behavior_mats_t[i], user_embedding)

                    # # add prompt
                    user_embedding_list[i] = torch.spmm(self.behavior_mats[i], item_embedding) + prompt
                    item_embedding_list[i] = torch.spmm(self.behavior_mats_t[i], user_embedding) + prompt



                    # if i == self.behaviorNum - 1:
                    #
                    #     user_embedding_list[i] = (user_embedding_list[i] @ prompt.t()) / (prompt @ prompt.t()) @ prompt
                    #     item_embedding_list[i] = (item_embedding_list[i] @ prompt.t()) / (prompt @ prompt.t()) @ prompt
                    # else:
                    #     pass

                else:
                    user_embedding_list[i] = torch.spmm(self.behavior_mats[i], item_embedding)
                    item_embedding_list[i] = torch.spmm(self.behavior_mats_t[i], user_embedding)
            else:
                u_emb_uu = self.userGNN(user_embedding , self.u_in[i])
                i_emb_ii = self.itemGNN(item_embedding, self.i_in[i], self.i_out[i])

                u_emb_ui = torch.spmm(self.behavior_mats[i], item_embedding)
                i_emb_ui = torch.spmm(self.behavior_mats_t[i], user_embedding)


                user_embedding_list[i] = torch.cat((u_emb_ui,u_emb_uu),-1) @ self.uu_w
                item_embedding_list[i] = torch.cat((i_emb_ii,i_emb_ui),-1) @ self.ii_w

        user_embeddings = torch.stack(user_embedding_list, dim=0)
        item_embeddings = torch.stack(item_embedding_list, dim=0)

        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w))
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w))



        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w))
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w))

        # return self.feat_drop(user_embedding), self.feat_drop(item_embedding),user_embeddings, item_embeddings
        return user_embedding, item_embedding,user_embeddings, item_embeddings

    def itemGNN(self, item_emb, i_in, i_out):
        in_neighbor = torch.spmm(i_in, item_emb)
        out_neighbor = torch.spmm(i_out, item_emb)

        x_in = self.relu((item_emb * in_neighbor) @ self.W1)
        x_out = self.relu((item_emb * out_neighbor) @ self.W2)

        in_score = torch.squeeze(torch.sum((x_in / math.sqrt(args.dim)), dim=1), 0)
        out_score = torch.squeeze(torch.sum((x_out / math.sqrt(args.dim)), dim=1), 0)
        score = self.softmax(torch.stack((in_score, out_score), dim=1))
        score_in = torch.unsqueeze(score[:, 0], dim=-1)
        score_out = torch.unsqueeze(score[:, 1], dim=-1)
        neighbor = in_neighbor * score_in + out_neighbor * score_out
        agg = torch.stack((item_emb, neighbor), dim=2)
        agg = torch.unsqueeze(agg, 1)
        out_conv = self.conv_layer(agg)
        emb = self.feat_drop(torch.squeeze(out_conv))

        return emb

    def userGNN(self, user_emb, u_in):
        neighbor_feature = torch.spmm(u_in, user_emb)

        agg = torch.stack((neighbor_feature, user_emb), dim=2)  # n x dim x 3
        agg = torch.unsqueeze(agg, 1)
        out_conv = self.conv_layer_user(agg)
        emb = self.feat_drop(torch.squeeze(out_conv))


        return emb



class graphDenoising(nn.Module):
    def __init__(self, dim, behavior):
        super(graphDenoising, self).__init__()
        self.dim = dim
        self.behavior = behavior
        # self.W = nn.Parameter(torch.Tensor(dim, dim))
        self.act = nn.Sigmoid()

        # init.xavier_uniform_(self.W)


    def forward(self, user_emb, pos_emb, neg_emb, beh_emb, prompt_emb):
        generated_graphs = [None] * self.behavior
        generated_graphsT = []
        for beh in range(self.behavior):
            if beh != self.behavior-1:
                w = beh_emb[beh]
            else:
                w = prompt_emb.squeeze()
            W = w.unsqueeze(1) @ w.unsqueeze(-1).t()
            ge_g_pos = self.act(torch.sum(user_emb[beh] @ W * pos_emb[beh], 1,keepdim=True))
            ge_g_neg = self.act(torch.sum(user_emb[beh] @ W * neg_emb[beh], 1,keepdim=True))
            ge_g = torch.cat((ge_g_pos,ge_g_neg),0)
            # ge_g = self.act(user_emb[beh] @ W @ item_emb[beh].t())
            generated_graphs[beh] = ge_g
        # for beh in range(self.behavior):
        #     ge_g = self.act(user_emb[beh] @ self.W @ item_emb[beh].t())
        #     generated_graphs[beh] = ge_g
        return generated_graphs, generated_graphsT
