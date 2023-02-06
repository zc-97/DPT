import numpy as np
import pickle
from tqdm import tqdm
import random
import torch
from Params import args
import datetime
from Model import myModel1

import torch.utils.data as dataloader

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
from datasets import process_ts


def pre_train(trainData, cf_train, testData, dim=16, epoch=300, batch_size=256, device='cuda:0'):
    tr_mats, tr_matsT, int_types, meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out, tr_labels = trainData
    labelP = np.squeeze(np.array(np.sum(tr_labels, axis=0)))

    item_in = []
    item_out = []
    user_in = []

    if args.prompt:
        if args.pattern:
            for beh in range(int_types):
                item_in.append(itemMats_in[beh].to(device))
                item_out.append(itemMats_out[beh].to(device))
                user_in.append(userMats_in[beh].to(device))

        else:
            item_in.append(torch.tensor([0,1]).to(device))
            item_out.append(torch.tensor([0,1]).to(device))
            user_in.append(torch.tensor([0,1]).to(device))
    else:
        for beh in range(int_types):
            item_in.append(itemMats_in[beh].to(device))
            item_out.append(itemMats_out[beh].to(device))
            user_in.append(userMats_in[beh].to(device))

    target_mat_for_test = tr_mats[-1].cpu().to_dense()

    tr_mats = [item.to(device) for item in tr_mats]
    tr_matsT = [item.to(device) for item in tr_matsT]
    target_beh = tr_mats[-1]
    user_num, item_num = target_beh.shape
    print('user_num:', user_num)
    print('item_num:', item_num)

    print('waiting for csr->dense')
    if args.prompt or args.denoise_tune:
        ori_graphs = []

    else:
        ori_graphs = [torch.tensor(item_.todense()).long().to(args.device) for item_ in np_mats]

    # train_loader = data_loader(meta_paths, batchSize=batch_size)

    test1 = (labelP, target_mat_for_test, testData)

    if args.just_test:
        if args.prompt and not args.denoise_tune:
            loadPath = r'./Model/' + args.dataset + r'/' + args.prompt_flag + '_deep_' + r'.pth'
        elif args.prompt and args.denoise_tune:
            loadPath = r'./Model/' + args.dataset + r'/' + args.tune_flag + r'.pth'

        params = torch.load(loadPath, map_location=torch.device(args.device))
        net = params['model']
        hit = params['hr']
        ndcg = params['ndcg']
        print(f'loaded model, hit:{hit}, ndcg:{ndcg}')
    else:
        net = model_prepare(int_types, item_in, item_num, item_out, tr_mats, tr_matsT, user_in, user_num, labelP,
                            target_mat_for_test, testData)

    epochHR, epochNDCG = [0] * 2

    print('test before train')
    net.eval()
    cnt, result_HR, result_NDCG, _ = cfTestEpoch(epochHR, epochNDCG, labelP, net, target_mat_for_test, testData)

    print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

    epoch_num = 0
    best_hr = 0
    best_ndcg = 0
    for epoch in range(epoch):
        epoch_num += 1
        cf_cost = 0
        rec_cost = 0
        brp_cost = 0
        reg_cost = 0

        '''训练'''
        print('start training')

        net.train()

        if args.prompt:
            epoch_loss = cfTrainEpoch(net, cf_train, int_types, ori_graphs=ori_graphs)
            cf_cost += epoch_loss

            print('\tcf_cost:\t%.3f' % cf_cost)
        else:
            epoch_loss, epoch_rec_loss, epoch_bpr_loss, epoch_reg_loss = cfTrainEpoch(net, cf_train, int_types,
                                                                                      ori_graphs=ori_graphs, target_mat_for_test=target_mat_for_test)
            cf_cost += epoch_loss
            rec_cost += epoch_rec_loss
            brp_cost += epoch_bpr_loss
            reg_cost += epoch_reg_loss

            print('\tcf_cost:\t%.3f' % cf_cost)
            print('\trec_cost:\t%.3f' % rec_cost)
            print('\tbrp_cost:\t%.3f' % brp_cost)
            print('\treg_cost:\t%.3f' % reg_cost)

        net.scheduler.step()
        net.eval()


        epochHR, epochNDCG = [0] * 2
        cnt, result_HR, result_NDCG, saved_embs = cfTestEpoch(epochHR, epochNDCG, labelP, net, target_mat_for_test,
                                                              testData)

        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        if result_HR > best_hr:
            best_hr = result_HR
            print(f'best_HR={best_hr},epoch={epoch}')
            user_embs, item_embs, user_embs_list, item_embs_list = saved_embs
            if args.wsdm:
                save_model(net, user_embs, user_embs_list, item_embs, item_embs_list, result_HR, result_NDCG, test1,
                           flag=args.tune_flag)
            else:
                if not args.prompt:
                    save_model(net, user_embs, user_embs_list, item_embs, item_embs_list, result_HR, result_NDCG, test1,
                               flag=args.pre_flag)

                if args.prompt and not args.denoise_tune:
                    if not args.deep:
                        save_model(net, user_embs, user_embs_list, item_embs, item_embs_list, result_HR, result_NDCG, test1,
                                   flag=args.prompt_flag)
                    elif args.deep:
                        save_model(net, user_embs, user_embs_list, item_embs, item_embs_list, result_HR, result_NDCG, test1,
                                   flag=args.prompt_flag + '_deep_')

                elif args.prompt and args.denoise_tune:
                    save_model(net, user_embs, user_embs_list, item_embs, item_embs_list, result_HR, result_NDCG, test1,
                               flag=args.tune_flag)

        if result_NDCG > best_ndcg:
            best_ndcg = result_NDCG
            print(f'best_NDCG={best_ndcg},epoch={epoch}')



def model_prepare(int_types, item_in, item_num, item_out, tr_mats, tr_matsT, user_in, user_num, labelP,
                  target_mat_for_test, testData):
    if args.prompt and not args.denoise_tune:
        # print('加载去噪后微调embedding layers的模型！！！！')
        print('直接 加载预训练模型 prompt tuning')
        # loadPath = r'./Model/' + args.dataset + r'/' + args.pre_flag + r'.pth'
        loadPath = r'./Model/' + args.dataset + r'/' + args.tune_flag + r'.pth'
        params = torch.load(loadPath, map_location=torch.device(args.device))
        pre_trained_net = params['model']
        pre_trained_net.behavior_mats = tr_mats
        pre_trained_net.behavior_matsT = tr_matsT

        # print('test loaded model')
        # epochHR, epochNDCG = [0] * 2
        #
        # pre_trained_net.eval()
        # cnt, result_HR, result_NDCG, _ = cfTestEpoch(epochHR, epochNDCG, labelP, pre_trained_net, target_mat_for_test, testData)
        #
        # print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        if args.pattern:
            pre_trained_net.i_in = item_in
            pre_trained_net.i_out = item_out
            pre_trained_net.u_in = user_in

        pre_dict = pre_trained_net.state_dict()

        if args.pattern:
            net = myModel1(userNum=user_num, itemNum=item_num, behavior=int_types, behavior_mats=tr_mats,
                           behavior_matsT=tr_matsT, i_in=item_in, i_out=item_out, u_in=user_in).to(args.device)
        else:
            # pre_item_in = pre_trained_net.i_in
            # pre_item_out = pre_trained_net.i_out
            # pre_user_in = pre_trained_net.u_in
            pre_item_in = torch.tensor([0,1]).to(args.device)
            pre_item_out = torch.tensor([0,1]).to(args.device)
            pre_user_in = torch.tensor([0,1]).to(args.device)
            net = myModel1(userNum=user_num, itemNum=item_num, behavior=int_types, behavior_mats=tr_mats,
                           behavior_matsT=tr_matsT, i_in=pre_item_in, i_out=pre_item_out, u_in=pre_user_in).to(
                args.device)

        net_dict = net.state_dict()

        for i,p in enumerate(net_dict):
            if i==0:
                net_dict[str(p)].copy_(torch.mean(pre_dict['beh_embedding.weight'][:-1], 0))
                # net_dict[str(p)].copy_(pre_dict[str(p)])

                # pass

            else:
                net_dict[str(p)].copy_(pre_dict[str(p)])
            # print(i, p)
            # if i>=38:
            #     pass
            # else:
            #     if i == 0:
            #         net_dict[str(p)].copy_(torch.mean(pre_dict['beh_embedding.weight'], 0))
            #     elif i == 7:
            #         net_dict[str(p)].copy_(torch.mean(pre_dict['gcn.behavior_embeddings'], 0))
            #     else:
            #         net_dict[str(p)].copy_(pre_dict[str(p)])
        # assert 1==2
        # assert 1==2
        #
        # for i, p in enumerate(net_dict):
        #     if i == 2 or i == 5:
        #         for beh in range(int_types):
        #             if beh != int_types-1:
        #                 net_dict[str(p)][beh].copy_(pre_dict[str(p)][beh])
        #             else:
        #                 net_dict[str(p)][-1].copy_(torch.mean(pre_dict[str(p)], 0))
        #     elif i >= 36:
        #         # 没有conv bias
        #         pass
        #     else:
        #         net_dict[str(p)].copy_(pre_dict[str(p)])

        if args.head:
            # update_para_name = ['beh_embedding.weight','gcn.behavior_embeddings','gcn.i_concatenation_w','gcn.u_concatenation_w','prompt.weight', 'gcn.prompt_embedding']
            update_para_name = ['gcn.i_concatenation_w','gcn.u_concatenation_w','prompt.weight', 'gcn.prompt_embedding']
            # head_name = ['gcn.i_concatenation_w','gcn.u_concatenation_w']
            # #
            # update_para_name = ['beh_embedding.weight', 'gcn.behavior_embeddings', 'gcn.layers.0.i_w',
            #                     'gcn.layers.0.u_w', 'gcn.layers.1.i_w', 'gcn.layers.1.u_w', 'gcn.layers.2.i_w',
            #                     'gcn.layers.2.u_w']
            # head_name = ['gcn.layers.0.i_w', 'gcn.layers.0.u_w', 'gcn.layers.1.i_w', 'gcn.layers.1.u_w',
            #              'gcn.layers.2.i_w', 'gcn.layers.2.u_w']
        else:
            # update_para_name = ['prompt.weight', 'gcn.prompt_embedding', 'beh_embedding.weight','gcn.behavior_embeddings']
            update_para_name = ['prompt.weight']
            head_name = []


        if args.noise_lambda > 0:
            for i, p in enumerate(net.named_parameters()):
                if 'bias' not in str(p[0]):
                    p[1].data += (torch.rand(p[1].data.size()).to(args.device) - 0.5) * args.noise_lambda * torch.std(
                        p[1].data)

        for i, p in enumerate(net.named_parameters()):
            if str(p[0]) not in update_para_name:
                p[1].requires_grad = False
            # if str(p[0]) == 'beh_embedding.weight':
            #     p[1][:-1].requires_grad = False

        # for i,p in enumerate(net.named_parameters()):
        #     print(i,p[0],p[1].requires_grad)
        # assert 1==2

        net.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                          lr=args.lr, weight_decay=args.opt_weight_decay)
        net.scheduler = torch.optim.lr_scheduler.CyclicLR(net.optimizer, args.opt_base_lr, args.opt_max_lr,
                                                          step_size_up=5, step_size_down=10, mode='triangular',
                                                          gamma=0.99, scale_fn=None, scale_mode='cycle',
                                                          cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
                                                          last_epoch=-1)

        del pre_trained_net

    elif args.prompt and args.denoise_tune:
        print('加载预训练模型,删除pattern部分！！！！')
        loadPath = r'./Model/' + args.dataset + r'/' + args.pre_flag + r'.pth'
        params = torch.load(loadPath, map_location=torch.device(args.device))
        pre_trained_net = params['model']

        if args.wsdm:
            pass
        else:
            pre_trained_net.behavior_mats = tr_mats
            pre_trained_net.behavior_matsT = tr_matsT

        if args.pattern:
            pre_trained_net.i_in = item_in
            pre_trained_net.i_out = item_out
            pre_trained_net.u_in = user_in

        pre_dict = pre_trained_net.state_dict()

        if args.pattern:
            net = myModel1(userNum=user_num, itemNum=item_num, behavior=int_types, behavior_mats=tr_mats,
                           behavior_matsT=tr_matsT, i_in=item_in, i_out=item_out, u_in=user_in).to(args.device)


        else:
            pre_item_in = pre_trained_net.i_in
            pre_item_out = pre_trained_net.i_out
            pre_user_in = pre_trained_net.u_in
            net = myModel1(userNum=user_num, itemNum=item_num, behavior=int_types, behavior_mats=tr_mats,
                           behavior_matsT=tr_matsT, i_in=pre_item_in, i_out=pre_item_out, u_in=pre_user_in).to(
                args.device)

        net_dict = net.state_dict()

        if args.wsdm:
            print('using wsdm-21 denoiseRec, lightGCN+TCE')
            pass
        else:
            frozen_para = ['user_embedding.weight', 'item_embedding.weight', 'beh_embedding.weight', 'gcn.behavior_embeddings', 'gcn.user_embedding', 'gcn.item_embedding']

            for i, p in enumerate(net_dict):
                if str(p) in frozen_para:
                    net_dict[str(p)].copy_(pre_dict[str(p)])
                # net_dict[str(p)].copy_(pre_dict[str(p)])


            for i, p in enumerate(net.named_parameters()):
                # if str(p[0]) not in update_para_name:
                if str(p[0]) in frozen_para:
                    p[1].requires_grad = False
            #
            if args.noise_lambda > 0:
                print('noise-tune:噪声量',args.noise_lambda)
                for name, p in net.named_parameters():
                #     if str(name) not in new_para:
                    net.state_dict()[name][:] += (torch.rand(p.size()).to(
                            args.device) - 0.5) * args.noise_lambda * torch.std(p)

            net.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                              lr=args.lr, weight_decay=args.opt_weight_decay)
            net.scheduler = torch.optim.lr_scheduler.CyclicLR(net.optimizer, args.opt_base_lr, args.opt_max_lr,
                                                              step_size_up=5, step_size_down=10, mode='triangular',
                                                              gamma=0.99, scale_fn=None, scale_mode='cycle',
                                                              cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
                                                              last_epoch=-1)

        del pre_trained_net


    else:
        net = myModel1(userNum=user_num, itemNum=item_num, behavior=int_types, behavior_mats=tr_mats,
                       behavior_matsT=tr_matsT, i_in=item_in, i_out=item_out, u_in=user_in).to(args.device)

    return net


def cfTestEpoch(epochHR, epochNDCG, labelP, net, target_mat_for_test, testData):
    with torch.no_grad():
        user_embs, item_embs, user_embs_list, item_emb_list, graphs = net()


        cnt = 0
        tot = 0
    for user, item_i in testData:
        user_compute, item_compute, user_item1, user_item100 = sampleTestBatch(user, item_i, target_mat_for_test,
                                                                               labelP)

        userEmbed = user_embs[user_compute]
        itemEmbed = item_embs[item_compute]

        pred_i = torch.sum(torch.mul(userEmbed, itemEmbed), dim=1)

        hit, ndcg = calcRes(torch.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)
        epochHR = epochHR + hit
        epochNDCG = epochNDCG + ndcg  #
        cnt += 1
        tot += user.shape[0]
    result_HR = epochHR / tot
    result_NDCG = epochNDCG / tot

    saved_embs = (user_embs, item_embs, user_embs_list, item_emb_list)

    return cnt, result_HR, result_NDCG, saved_embs


if args.prompt and not args.denoise_tune:
    print('prompt tuning')


    def cfTrainEpoch(net, train_loader, behavior_num, ori_graphs):
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()
        time = datetime.datetime.now()
        print("end_ng_samp:  ", time)

        epoch_loss = 0

        # -----------------------------------------------------------------------------------
        behavior_loss_list = [None]

        user_id_list = [None]
        item_id_pos_list = [None]
        item_id_neg_list = [None]

        # ----------------------------------------------------------------------------------
        cnt = 0
        for user, item_i, item_j in tqdm(train_loader):
            user = user.long().cuda()
            user_embed, item_embed, user_embeds, item_embeds, graphs = net()

            for index in range(len(behavior_loss_list)):
                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]

                user_id_list[index] = user[not_zero_index].long().cuda()
                item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                userEmbed = user_embed[user_id_list[index]]
                posEmbed = item_embed[item_id_pos_list[index]]
                negEmbed = item_embed[item_id_neg_list[index]]

                pred_i, pred_j = 0, 0
                pred_i, pred_j = innerProduct(userEmbed, posEmbed, negEmbed)

                behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()

            for i in range(len(behavior_loss_list)):
                behavior_loss_list[i] = (behavior_loss_list[i]).sum()

            bprloss = sum(behavior_loss_list) / len(behavior_loss_list)

            regLoss = (torch.norm(userEmbed) ** 2 + torch.norm(posEmbed) ** 2 + torch.norm(negEmbed) ** 2) * (args.reg)

            loss = (bprloss+regLoss) / args.batch

            epoch_loss = epoch_loss + loss.item()

            net.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            net.optimizer.step()
            cnt += 1
        return epoch_loss

elif args.prompt and args.denoise_tune:
    print('噪声微调只用auxiliary behavior， 不需要还原图')


    def cfTrainEpoch(net, train_loader, behavior_num, ori_graphs):
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()

        time = datetime.datetime.now()
        print("end_ng_samp:  ", time)

        epoch_loss = 0

        # -----------------------------------------------------------------------------------
        behavior_loss_list = [None] * (behavior_num)


        user_id_list = [None] * (behavior_num)
        item_id_pos_list = [None] * (behavior_num)
        item_id_neg_list = [None] * (behavior_num)

        # ----------------------------------------------------------------------------------
        cnt = 0
        for user, item_i, item_j in tqdm(train_loader):

            item_list = [None] * (behavior_num)
            user_set = torch.tensor(list(set(user.tolist())))
            for i in range((behavior_num)):
                item_set = list(set(item_i[i].tolist()))
                try:
                    item_set.remove(-1)
                except:
                    pass

                item_list[i] = torch.tensor(item_set)

            user = user.long().cuda()
            user_embed, item_embed, user_embeds, item_embeds, graphs = net(user_set, item_list, is_denoise=True)

            # from numba import njit

            for index in range((behavior_num)):
                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]

                user_id_list[index] = user[not_zero_index].long().cuda()
                item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                userEmbed = user_embed[user_id_list[index]]
                posEmbed = item_embed[item_id_pos_list[index]]
                negEmbed = item_embed[item_id_neg_list[index]]

                pred_i, pred_j = 0, 0
                pred_i, pred_j = innerProduct(userEmbed, posEmbed, negEmbed)

                behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()


            for i in range((behavior_num)):
                if args.wsdm:
                    def drop_rate_schedule(iteration):

                        drop_rate = np.linspace(0, 0.2, 10000)
                        if iteration < 10000:
                            return drop_rate[iteration]
                        else:
                            return 0.2

                    def TCE(loss_list_, rate_):
                        flatten_loss = loss_list_.flatten()
                        ind_sorted_ = np.argsort(flatten_loss.cpu().data)
                        loss_sorted_ = flatten_loss[ind_sorted_]
                        remember_rate_ = 1 - rate_
                        num_remember_ = int(remember_rate_ * len(loss_sorted_))
                        ind_update_ = ind_sorted_[:num_remember_]
                        loss_update_ = flatten_loss[ind_update_]
                        return loss_update_.sum()

                    behavior_loss_list[i] = TCE(behavior_loss_list[i], drop_rate_schedule(cnt))
                else:
                    behavior_loss_list[i] = (behavior_loss_list[i]).sum()

            bprloss = sum(behavior_loss_list) / len(behavior_loss_list)
            regLoss = (torch.norm(userEmbed) ** 2 + torch.norm(posEmbed) ** 2 + torch.norm(negEmbed) ** 2)

            loss = (bprloss + args.reg * regLoss) / args.batch

            epoch_loss = epoch_loss + loss.item()


            net.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            net.optimizer.step()

            cnt += 1

        return epoch_loss
else:
    print('预训练只用auxiliary behaviors')

    def cfTrainEpoch(net, train_loader, behavior_num, ori_graphs, target_mat_for_test):
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()
        time = datetime.datetime.now()
        print("end_ng_samp:  ", time)

        epoch_loss = 0
        epoch_rec_loss = 0
        epoch_bpr_loss = 0
        epoch_reg_loss = 0
        # behavior_nums = behavior_num + 1
        # -----------------------------------------------------------------------------------
        behavior_loss_list = [None] * (behavior_num)
        reconstruction_loss_list = [None] * (behavior_num)

        loss_fuc = torch.nn.CrossEntropyLoss()

        user_id_list = [None] * (behavior_num)
        item_id_pos_list = [None] * (behavior_num)
        item_id_neg_list = [None] * (behavior_num)

        # ----------------------------------------------------------------------------------
        cnt = 0
        for user, item_i, item_j in tqdm(train_loader):
            user = user.long().to(args.device)
            item_i = [item.long().to(args.device) for item in item_i]
            item_j = [item.long().to(args.device) for item in item_j]
            item_list = [None] * (behavior_num)
            user_set = torch.tensor(list(set(user.tolist())))
            # for i in range((behavior_num)):
            #     item_set = list(set(item_i[i].tolist()))
            #     try:
            #         item_set.remove(-1)
            #     except:
            #         pass
            #
            #     item_list[i] = torch.tensor(item_set)

            ori_graph_labels = [None] * (behavior_num)

            for i in range((behavior_num)):
                # if i == behavior_num-1:
                #     ori_graph_labels[i] = torch.flatten(torch.from_numpy(target_mat_for_test[user_set, :][:, item_list[i]].long()).to(args.device))
                # else:
                ori_graph = ori_graphs[i]
                pos = ori_graph[(user,item_i[i])]
                neg = ori_graph[(user,item_j[i])]
                ori_graph_labels[i] = torch.cat((pos,neg),0).unsqueeze(-1)
                # ori_graph_labels[i] = torch.flatten(ori_graphs[i][user_set, :][:, item_list[i]])

            user = user.long().to(args.device)
            user_embed, item_embed, user_embeds, item_embeds, graphs = net(user, item_i, item_j, is_denoise=True)
            for i in range((behavior_num)):
                label = ori_graph_labels[i]
                reconstruction_graph = graphs[i]

                neg_ = 1 - reconstruction_graph
                decoder_res = torch.stack((neg_, reconstruction_graph), 1)
                reconstruction_loss_list[i] = loss_fuc(decoder_res, label)

            for index in range((behavior_num)):
                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]

                user_id_list[index] = user[not_zero_index].long().cuda()
                item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                userEmbed = user_embed[user_id_list[index]]
                posEmbed = item_embed[item_id_pos_list[index]]
                negEmbed = item_embed[item_id_neg_list[index]]


                pred_i, pred_j = 0, 0
                pred_i, pred_j = innerProduct(userEmbed, posEmbed, negEmbed)

                behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()

            rec_loss = sum(reconstruction_loss_list) / len(reconstruction_loss_list)

            for i in range((behavior_num)):
                behavior_loss_list[i] = (behavior_loss_list[i]).sum()

            bprloss = sum(behavior_loss_list) / len(behavior_loss_list)
            regLoss = (torch.norm(userEmbed) ** 2 + torch.norm(posEmbed) ** 2 + torch.norm(negEmbed) ** 2)

            beh_reg = torch.norm(net.beh_embedding.weight) ** 2 + torch.norm(net.prompt.weight) ** 2

            loss = (bprloss + args.reg * regLoss) / args.batch + rec_loss / 2 + (beh_reg * args.reg / behavior_num)

            epoch_loss = epoch_loss + loss.item()

            epoch_rec_loss = epoch_rec_loss + (rec_loss / 2).item()
            epoch_bpr_loss = epoch_bpr_loss + (bprloss / args.batch).item()
            epoch_reg_loss = epoch_reg_loss + (args.reg * regLoss / args.batch).item()

            net.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            net.optimizer.step()

            cnt += 1

        return epoch_loss, epoch_rec_loss, epoch_bpr_loss, epoch_reg_loss


def calcRes(pred_i, user_item1, user_item100):  # [6144, 100] [6144] [6144, (ndarray:(100,))]

    hit = 0
    ndcg = 0

    for j in range(pred_i.shape[0]):

        _, shoot_index = torch.topk(pred_i[j], args.shoot)
        shoot_index = shoot_index.cpu()
        shoot = user_item100[j][shoot_index]
        shoot = shoot.tolist()

        if type(shoot) != int and (user_item1[j] in shoot):
            hit += 1
            ndcg += np.reciprocal(np.log2(shoot.index(user_item1[j]) + 2))
        elif type(shoot) == int and (user_item1[j] == shoot):
            hit += 1
            ndcg += np.reciprocal(np.log2(0 + 2))

    return hit, ndcg  # int, float


def innerProduct(u, i, j):
    pred_i = torch.sum(torch.mul(u, i), dim=1)
    pred_j = torch.sum(torch.mul(u, j), dim=1)
    return pred_i, pred_j


def sampleTestBatch(batch_user_id, batch_item_id, trainMat_target, labelP):
    trainMat_target = trainMat_target.detach().cpu()
    batch = len(batch_user_id)  # e.g., 8K
    tmplen = (batch * 100)  # e.g. 800K

    sub_trainMat = trainMat_target[batch_user_id].numpy()  # 从交互记录中取出了batch_user_id对应行
    user_item1 = batch_item_id
    user_compute = [None] * tmplen
    item_compute = [None] * tmplen
    user_item100 = [None] * (batch)

    cur = 0
    for i in range(batch):
        pos_item = user_item1[i]
        negset = np.reshape(np.argwhere(sub_trainMat[i] == 0), [-1])
        pvec = labelP[negset]
        pvec = pvec / np.sum(pvec)

        random_neg_sam = np.random.permutation(negset)[:99]
        user_item100_one_user = np.concatenate((random_neg_sam, np.array([pos_item])))
        user_item100[i] = user_item100_one_user

        for j in range(100):
            user_compute[cur] = batch_user_id[i]
            item_compute[cur] = user_item100_one_user[j]
            cur += 1

    return user_compute, item_compute, user_item1, user_item100


def setRandomSeed():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)


def save_model(model, user_embed, user_embeds, item_embed, item_embeds, best_hr, ndcg, test_data_tuple, flag):

    print(model.item_embedding.weight[0][:2])
    savePath = r'./Model/' + args.dataset + r'/' + flag + r'.pth'
    print(f'u_emb.shape:{user_embed.shape}, i_emb.shape:{item_embed.shape}')
    if args.prompt and args.denoise_tune:
        params = {
            'model': model,
            'user_embed': user_embed,
            'item_embed': item_embed,
            'hr':best_hr,
            'ndcg':ndcg,
        }
    else:
        params = {
            'model': model,
            'user_embed': user_embed,
            'user_embeds': user_embeds,
            'item_embed': item_embed,
            'item_embeds': item_embeds,
            'hr':best_hr,
            'ndcg':ndcg,
        }
    torch.save(params, savePath)
    print('model saved')



def load_model(loadPath, labelP, target_mat_for_test, testData):
    params = torch.load(loadPath)
    model = params['model']
    user_emb = params['user_embed']
    user_embs = params['item_embed']
    item_emb = params['user_embeds']
    item_embs = params['item_embeds']
    load_hr = params['hr']
    load_ndcg = params['ndcg']
    print(model.item_embedding.weight[0][:2])
    epochHR, epochNDCG = [0] * 2
    model.eval()
    cnt, hr, ndcg, saved_embs = cfTestEpoch(epochHR, epochNDCG, labelP, model, target_mat_for_test, testData)

    return model, user_emb, item_emb, user_embs, item_embs, load_hr, load_ndcg, hr, ndcg



if __name__ == '__main__':
    setRandomSeed()
    print(args)
    print('start data pre-process')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    datas = process_ts.load_data()
    print('over!')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('---------train-------------')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # behavior_mats, trnMats, trnMatsT, te_int, tr_label, te_users, len(
    #     behaviors), meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out = datas

    np_mats, tr_mats, tr_matsT, tr_label, cf_train_data, test_data, int_types, meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out = datas
    torch.cuda.empty_cache()

    pre_train_data = (
        tr_mats, tr_matsT, int_types, meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out, tr_label)

    pre_train(pre_train_data, cf_train_data, test_data, dim=16, epoch=500, batch_size=128, device=args.device)

    # np_mats, tr_mats, tr_matsT, te_int, tr_label, te_users, int_types = datas
