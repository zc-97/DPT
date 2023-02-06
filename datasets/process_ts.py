import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Params import args
import scipy.sparse as sp
import pandas as pd
import torch
import torch.utils.data as data
import pickle
import numpy as np
import scipy.sparse as sp
from math import ceil
import datetime
from scipy import sparse
from tqdm import tqdm
from Params import args
import Utils.graph_util

if args.dataset == 'yelp' or args.dataset == 'Yelp':
    # predir = 'WWW23-MB-DGD/datasets/Yelp/'
    predir = 'datasets/Yelp/'
    behaviors = ['tip', 'neg', 'neutral', 'pos']
    user_num = 19800
    item_num = 22734
    # USER 19800 ITEM 22734

elif args.dataset == 'IJCAI_15' or args.dataset == 'IJCAI':
    # predir = 'WWW23-MB-DGD/datasets/Yelp/'
    predir = 'datasets/IJCAI_15/'
    behaviors = ['click', 'fav', 'cart', 'buy']
    user_num = 17435
    item_num = 35920

elif args.dataset == 'Tmall' or args.dataset == 'tmall':
    predir = 'datasets/Tmall/'
    behaviors = ['pv', 'fav', 'cart', 'buy']
    user_num = 31882
    item_num = 31232
    # USER 17435 ITEM 35920
# else:
#     predir = 'None'
#     behaviors = []
#     print('have not implement')
# tr_file = predir + 'trn_'
# te_file = predir + 'tst_'


# predir = 'WWW23-MB-DGD/datasets/Yelp/'
# behaviors = ['tip', 'neg', 'neutral', 'pos']
tr_file = predir + 'trn_'
te_file = predir + 'tst_'


def load_data():

        ori_mats = list()
        trnMats = list()  # 所有behavior matrix组成的list,每一个都是torch.Tensor

        behavior_mats = list()
        trnMatsT = list()

        seq_trnMats = list()
        seq_trnMatsT = list()
        for i in range(len(behaviors)):
            beh = behaviors[i]
            path = tr_file + beh
            with open(path, 'rb') as fs:
                mat = pickle.load(fs)
                ori_mats.append(mat)
                mat = (mat != 0) * 1
            behavior_mats.append(mat)
            if args.target == 'click':
                tr_label = (mat if i == 0 else 1 * (tr_label + mat != 0))
                user_num, item_num = mat.shape
            elif args.target == 'buy' and i == len(behaviors) - 1:
                tr_label = 1 * (mat != 0)
                user_num, item_num = mat.shape
                target_Mat = mat
                print(user_num, item_num)

            trnMatsT.append(matrix_to_tensor(mat.T))
            mat = matrix_to_tensor(mat)
            trnMats.append(mat)

            seq_trnMats.append(matrix_to_tensor(ori_mats[i]))
            seq_trnMatsT.append(matrix_to_tensor(ori_mats[i].T))
        if args.wsdm:
            meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out = (0, 0, 0, 0, 0)
        elif not args.wsdm:
            if args.prompt:
                time1 = datetime.datetime.now()
                print('加载去噪图', time1)
                trnMats = list()  # 所有behavior matrix组成的list,每一个都是torch.Tensor
                trnMatsT = list()
                seq_trnMats = list()
                seq_trnMatsT = list()

                ori_int_nums = [behavior_mats[i].sum() for i in range(len(behaviors))]

                behavior_mats, seq_mats = loadModel(behavior_mats, ori_mats)

                ge_int_nums = [behavior_mats[i].sum() for i in range(len(behaviors))]

                for beh in range(len(behaviors)):
                    print(
                        f'ori_int:{ori_int_nums[beh]},ge_int:{ge_int_nums[beh]},noise_percent:{1 - (ge_int_nums[beh] / ori_int_nums[beh])}')

                for beh in range(len(behaviors)):
                    mat = behavior_mats[beh]
                    trnMatsT.append(matrix_to_tensor(mat.T))
                    mat = matrix_to_tensor(mat)
                    trnMats.append(mat)

                    seq_trnMats.append(matrix_to_tensor(seq_mats[beh]))
                    seq_trnMatsT.append(matrix_to_tensor(seq_mats[beh].T))

                time2 = datetime.datetime.now()

                print('加载去噪图完毕', time2)

        # target_matrix = trnMats[-1]

        'trnMats, maxTime = timeProcess(trnMats)'
        try:
            print('userNum, itemNum:', (user_num, item_num))
        except:
            print('ERROR')
            assert 1 == 2
        if args.wsdm:
            pass
        elif not args.wsdm:
            if args.prompt or args.denoise_tune:
                if args.pattern:
                    try:
                        print('try to open loaded files (graph denoising)')
                        with open(predir + 'temp/' + args.tune_flag + '_graphs.pkl', 'rb') as f:
                            print('file exist (graph denoising)')
                            temp_data = pickle.load(f)
                        meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out = temp_data
                    except:
                        print('file not exist (graph denoising)')
                        meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out = get_trn_meta_path(
                            seq_trnMats, seq_trnMatsT)
                        temp_data = (meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out)
                        with open(predir + 'temp/' + args.tune_flag + '_graphs.pkl', 'wb') as f:
                            pickle.dump(temp_data, f)
                else:
                    meta_paths = []
                    itemMats_in = []
                    itemMats_out = []
                    userMats_in = []
                    userMats_out = []
            else:
                try:
                    print('try to open loaded files')
                    with open(predir + 'temp/temp_data.pkl', 'rb') as f:
                        print('file exist')
                        temp_data = pickle.load(f)
                    meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out = temp_data
                except:
                    print('file not exist')
                    meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out = get_trn_meta_path(seq_trnMats,
                                                                                                         seq_trnMatsT)
                    temp_data = (meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out)
                    with open(predir + 'temp/temp_data.pkl', 'wb') as f:
                        pickle.dump(temp_data, f)

        path = te_file + 'int'
        with open(path, 'rb') as fs:
            te_int = pickle.load(fs)

        # test_user = np.array([idx for idx, i in enumerate(te_int) if i is not None])
        # test_item = np.array([i for idx, i in enumerate(te_int) if i is not None])
        # test_data = np.hstack((test_user.reshape(-1,1), test_item.reshape(-1,1))).tolist()
        # test_dataset = RecDataset(test_data, item_num, trnMats[-1], 0, False)

        # --------------------- CF-based train loader--------------------------
        train_u, train_v = behavior_mats[-1].nonzero()
        train_data = np.hstack((train_u.reshape(-1, 1), train_v.reshape(-1, 1))).tolist()
        train_dataset = RecDataset_beh(behaviors, train_data, item_num, behavior_mats, True)
        train_loader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                                              num_workers=4,
                                                              pin_memory=True)

        # --------------------- CF-based test loader --------------------------
        test_user = np.array([idx for idx, i in enumerate(te_int) if i is not None])
        test_item = np.array([i for idx, i in enumerate(te_int) if i is not None])
        # tstUsrs = np.reshape(np.argwhere(data!=None), [-1])
        test_data = np.hstack((test_user.reshape(-1, 1), test_item.reshape(-1, 1))).tolist()
        # testbatch = np.maximum(1, args.batch * args.sampNum

        test_dataset = RecDataset(test_data, item_num, target_Mat, 0, False)
        test_loader = torch.utils.data.dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                                                             num_workers=4,
                                                             pin_memory=True)

        # te_stat = (te_int != None)
        # te_users = np.reshape(np.argwhere(te_stat != False), [-1])
        return behavior_mats, trnMats, trnMatsT, tr_label, train_loader, test_loader, len(
            behaviors), meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out



def loadModel(sparse_trn_mats, ori_mats):
    # denoising_test_2
    # ModelName = self.modelName
    # loadPath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
    loadPath = r'./Model/' + args.dataset + r'/' + args.pre_flag + r'.pth'
    params = torch.load(loadPath, map_location='cpu')

    with torch.no_grad():
        new_trn_mats = [None] * len(behaviors)
        new_seq_mats = [None] * len(behaviors)

        model = params['model'].cpu()
        user_embed = params['user_embed'].cpu()
        item_embed = params['item_embed'].cpu()
        user_embeds = params['user_embeds'].cpu()
        item_embeds = params['item_embeds'].cpu()

        # generated_graphs,_ = model.denoising(user_embed,[item_embed]*len(behaviors))
        generated_graphs = [None] * (len(behaviors) - 1)
        act = torch.nn.Sigmoid()
        for beh in range(len(behaviors) - 1):
            beh_emb = model.beh_embedding.weight[beh]

            beh_w = beh_emb.unsqueeze(1) @ beh_emb.unsqueeze(-1).t()

            generated_graphs[beh] = act(user_embed @ beh_w @ item_embed.t())


        for beh in range(len(behaviors)):
            if args.dataset == 'IJCAI_15':
                if beh != (len(behaviors) - 1) and beh != (len(behaviors) - 2):
                    # g = generated_graphs[beh].clone()
                    g = generated_graphs[beh]
                    if beh == 0:

                        g[g >= args.gumbel] = 1
                        g[g < args.gumbel] = 0
                    else:
                        g[g >= args.gumbel] = 1
                        g[g < args.gumbel] = 0
                    ori_g = torch.from_numpy(sparse_trn_mats[beh].todense()).to(g.device)
                    seq_g = torch.from_numpy(ori_mats[beh].todense()).to(g.device)  # time stamp

                    new_g = g * ori_g
                    new_seq_g = new_g * seq_g

                    if args.case:
                        user = torch.tensor(range(ori_g.shape[0]))

                    new_trn_mats[beh] = sp.csr_matrix(new_g.long().cpu().numpy())
                    new_seq_mats[beh] = sp.csr_matrix(new_seq_g.long().cpu().numpy())
                else:
                    new_trn_mats[beh] = sparse_trn_mats[beh]
                    new_seq_mats[beh] = ori_mats[beh]
            elif args.dataset == 'Tmall':
                if beh != (len(behaviors) - 1):
                    # g = generated_graphs[beh].clone()
                    g = generated_graphs[beh]
                    if beh == 0:
                        # g[g >= 0.44] = 1
                        # g[g < 0.44] = 0

                        g[g >= args.gumbel] = 1
                        g[g < args.gumbel] = 0
                    elif beh == (len(behaviors) - 2):
                        # g[g >= 0.46] = 1
                        # g[g < 0.46] = 0

                        g[g >= args.gumbel] = 1
                        g[g < args.gumbel] = 0
                    else:
                        g[g >= args.gumbel] = 1
                        g[g < args.gumbel] = 0
                    ori_g = torch.from_numpy(sparse_trn_mats[beh].todense()).to(g.device)
                    seq_g = torch.from_numpy(ori_mats[beh].todense()).to(g.device)  # time stamp

                    new_g = g * ori_g
                    new_seq_g = new_g * seq_g

                    new_trn_mats[beh] = sp.csr_matrix(new_g.long().cpu().numpy())
                    new_seq_mats[beh] = sp.csr_matrix(new_seq_g.long().cpu().numpy())
                else:
                    new_trn_mats[beh] = sparse_trn_mats[beh]
                    new_seq_mats[beh] = ori_mats[beh]

        del params
        torch.cuda.empty_cache()

    return new_trn_mats, new_seq_mats


def get_trn_meta_path(trnMats, trnMatsT):
    max_len = 0
    meta_path = {}
    # user:(item_seq, beh_seq, mask_seq)
    # item_mask = item_num
    # beh_mask = beh_num
    # mask_seq = 1/0
    print('create mate-paths')
    for user in tqdm(range(user_num), total=user_num, ncols=100):
        item_list = []
        time_list = []
        beh_list = []
        for i in range(len(behaviors)):
            beh = behaviors[i]
            path = tr_file + beh
            with open(path, 'rb') as fs:
                mat = pickle.load(fs)
            # mat = trnMats[i]
            item = mat[user].nonzero()[1]  # np.ndarray

            time = mat[user, item].data  # np.ndarray

            item_list.extend(item)
            time_list.extend(time)
            assert len(item) == len(time)
            beh_list.extend([i] * len(item))
        df = pd.DataFrame(columns=['iid', 'time', 'beh'], index=range(len(item_list)))
        df['iid'] = item_list
        df['time'] = time_list
        df['beh'] = beh_list
        df1 = df.sort_values(['time'])
        item_sequence = df1['iid'].tolist()
        beh_sequence = df1['beh'].tolist()
        # item_set = df1['iid'].unique().tolist()
        # beh_set = df1['beh'].unique().tolist()

        seq_len = len(item_sequence)
        if seq_len >= max_len:
            max_len = seq_len
        # sequence = [item_sequence, beh_sequence, item_set, beh_set]
        sequence = [item_sequence, beh_sequence]
        meta_path[user] = sequence

    itemMats_in, itemMats_out, userMats_in, userMats_out = create_graphs(meta_path, trnMats, trnMatsT, is_padding=False)

    print('padding meta-paths')
    user_id_seq = torch.tensor(range(user_num))
    item_seqs_list = []
    beh_seqs_list = []
    mask_seqs_list = []
    item_sets_list = []
    beh_sets_list = []
    for user in tqdm(range(user_num), total=user_num, ncols=100):
        item_seq = meta_path[user][0]
        beh_seq = meta_path[user][1]

        length = len(item_seq)
        fix = max_len - len(item_seq)

        mask_seq = [1] * length + [0] * fix
        item_seq.extend([item_num] * fix)
        beh_seq.extend([len(behaviors)] * fix)

        item_seqs_list.append(np.array(item_seq))
        beh_seqs_list.append(np.array(beh_seq))
        mask_seqs_list.append(np.array(mask_seq))
    item_seqs_list = np.array(item_seqs_list)
    beh_seqs_list = np.array(beh_seqs_list)
    mask_seqs_list = np.array(mask_seqs_list)
    meta_paths = (
        user_id_seq,
        torch.from_numpy(item_seqs_list),
        torch.from_numpy(beh_seqs_list),
        torch.from_numpy(mask_seqs_list)
    )
    # mask_seq1 = np.array(mask_seq)
    # item_seq1 = np.array(item_seq)
    # beh_seq1 = np.array(beh_seq)
    # user_seq1 = np.array([user])  # [u_i]
    # meta_path[user] = (user_seq1, item_seq1, beh_seq1, mask_seq1)

    meta_paths = 0

    return meta_paths, itemMats_in, itemMats_out, userMats_in, userMats_out


def create_graphs(meta_path, trnMats, trnMatsT, is_padding=True):
    print('create_graphs')
    item_graph_list = []
    user_graph_list = []
    if is_padding:
        for i in range(len(behaviors)):
            item_graph = np.zeros((item_num + 1, item_num + 1), dtype=int)  # 包含了mask
            user_graph = np.zeros((user_num, user_num), dtype=int)  # user没有mask
            item_graph_list.append(item_graph)
            user_graph_list.append(user_graph)
    else:
        for i in range(len(behaviors)):
            item_graph = np.zeros((item_num, item_num), dtype=int)  # 包含了mask
            user_graph = np.zeros((user_num, user_num), dtype=int)  # user没有mask
            item_graph_list.append(item_graph)
            user_graph_list.append(user_graph)

    item_in, item_out = create_item_graphs(meta_path, item_graph_list)
    user_in, user_out = create_user_graphs(trnMats, trnMatsT, user_graph_list)

    return item_in, item_out, user_in, user_out


def create_item_graphs(meta_path, item_graph_list):
    print('create item graphs')
    in_list = []
    out_list = []
    for user in tqdm(range(user_num), total=user_num, ncols=100):
        item_seq = meta_path[user][0]
        beh_seq = meta_path[user][1]

        for i in range(len(item_seq) - 1):

            for j in range(i + 1, len(item_seq)):
                assert j < len(item_seq)

                item_i = item_seq[i]

                item_j = item_seq[j]
                beh_j = beh_seq[j]

                # e^t_{ij} += 1
                item_graph_list[beh_j][item_i][item_j] += 1
    print('convert to degree mat')
    for beh in range(len(behaviors)):
        in_g, out_g = get_degree_matrix(item_graph_list[beh])
        in_list.append(in_g)
        out_list.append(out_g)
    return in_list, out_list


def create_user_graphs(behavior_graphs, behavior_graphs_t, user_graph_list):
    with torch.no_grad():

        print('create user graphs')
        in_list = []
        out_list = []
        shapes = 0
        for beh in range(len(behaviors)):
            beh_graph = behavior_graphs[beh]
            beh_graph_t = behavior_graphs_t[beh]

            user_graph_list[beh] = (beh_graph @ beh_graph_t.to_dense()).data
            # user_graph_list[beh] = beh_graph.matmul(beh_graph_t)

            shapes = beh_graph.shape[0]

        total = torch.zeros((shapes, shapes)).to(args.device)
        for beh in range(len(behaviors)):
            total += user_graph_list[beh]

        # cup_uid, cup_iid = torch.nonzero(total, as_tuple=True)
        # cup_num = total[cup_uid, cup_iid]
        for beh in range(len(behaviors)):
            cap_uid, cap_iid = torch.nonzero(user_graph_list[beh], as_tuple=True)
            cap_num = user_graph_list[beh][cap_uid, cap_iid]
            cup_num = total[cap_uid, cap_iid]
            user_graph_list[beh][cap_uid, cap_iid] = cap_num / cup_num

            in_g, out_g = get_degree_matrix(user_graph_list[beh].detach().cpu().numpy())
            in_list.append(in_g)
            out_list.append(out_g)

        return in_list, out_list


def matrix_to_tensor(cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
    values = torch.from_numpy(cur_matrix.data)
    shape = torch.Size(cur_matrix.shape)

    if torch.cuda.is_available():
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(args.device)
    else:
        assert 1 == 2
        return 0


def bool_numpy(numpy_array):
    numpy_array_1 = numpy_array.copy()
    numpy_array_1[numpy_array_1 == 0.] = 1
    return numpy_array_1


def get_degree_matrix(adj_matrix):
    '''
    A = [ 1, 2, 2,
          0, 4, 6,
          1, 0, 0 ]

    in = [ 0.5, 0.0, 0.5,
           0.3,  0.7,  0.0,
           1.0,  0.0,  0.0 ]

    out = [ 0.2, 0.4, 0.4,
            0.0  0.4  0.6,
            1.0  0.0  0.0 ]

    NOTE: E = AE --> E \in R^{n X d}
    '''
    d = np.shape(adj_matrix)[0]
    row_temp = np.sum(adj_matrix, axis=0)
    row = bool_numpy(row_temp)
    row = np.reshape(row, (1, d))
    col_temp = np.sum(adj_matrix, axis=1)
    col = bool_numpy(col_temp)
    col = np.reshape(col, (d, 1))
    a_out = adj_matrix / col
    a_in = adj_matrix / row
    a_in = a_in.T

    a_in = dense2sparse(a_in)
    a_out = dense2sparse(a_out)

    # a_out = torch.from_numpy(a_out)
    # a_in = torch.from_numpy(a_in)
    return a_in, a_out


def dense2sparse(_matrix):
    a_ = sparse.coo_matrix(_matrix)
    v1 = a_.data
    indices = np.vstack((a_.row, a_.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = a_.shape
    if torch.cuda.is_available():
        sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(torch.float32).to(args.device)
    else:
        sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_matrix


class RecDataset(data.Dataset):
    def __init__(self, data, num_item, train_mat=None, num_ng=1, is_training=True):
        super(RecDataset, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.train_mat = train_mat
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        dok_trainMat = self.train_mat.todok()
        length = self.data.shape[0]
        self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)

        for i in range(length):  #
            uid = self.data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in dok_trainMat:
                while (uid, iid) in dok_trainMat:
                    iid = np.random.randint(low=0, high=self.num_item)
                    self.neg_data[i] = iid
                self.neg_data[i] = iid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.data[idx][1]

        if self.is_training:
            neg_data = self.neg_data
            item_j = neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i

    def getMatrix(self):
        pass

    def getAdj(self):
        pass

    def sampleLargeGraph(self):

        def makeMask():
            pass

        def updateBdgt():
            pass

        def sample():
            pass

    def constructData(self):
        pass


class RecDataset_beh(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):
        super(RecDataset_beh, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0]
        self.neg_data = [None] * self.length
        self.pos_data = [None] * self.length

    if args.prompt and not args.denoise_tune:
        def ng_sample(self):
            assert self.is_training, 'no need to sampling when testing'
            for i in range(self.length):
                self.pos_data[i] = [None]
                self.neg_data[i] = [None]
            train_u, train_v = self.behaviors_data[-1].nonzero()
            beh_dok = self.behaviors_data[-1].todok()
            set_pos = np.array(list(set(train_v)))

            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

            for i in range(self.length):

                uid = self.data[i][0]
                iid_neg = self.neg_data[i][0] = self.neg_data_index[i]
                iid_pos = self.pos_data[i][0] = self.pos_data_index[i]

                if (uid, iid_neg) in beh_dok:
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)
                        self.neg_data[i][0] = iid_neg
                    self.neg_data[i][0] = iid_neg

                self.pos_data[i][0] = train_v[i]
    else:

        def ng_sample(self):
            assert self.is_training, 'no need to sampling when testing'

            for i in range(self.length):
                self.neg_data[i] = [None] * len(self.beh)
                self.pos_data[i] = [None] * len(self.beh)

            for index in range(len(self.beh)):

                train_u, train_v = self.behaviors_data[index].nonzero()
                beh_dok = self.behaviors_data[index].todok()

                set_pos = np.array(list(set(train_v)))

                self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
                self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

                for i in range(self.length):  #

                    uid = self.data[i][0]
                    iid_neg = self.neg_data[i][index] = self.neg_data_index[i]
                    iid_pos = self.pos_data[i][index] = self.pos_data_index[i]

                    if (uid, iid_neg) in beh_dok:
                        while (uid, iid_neg) in beh_dok:
                            iid_neg = np.random.randint(low=0, high=self.num_item)
                            self.neg_data[i][index] = iid_neg
                        self.neg_data[i][index] = iid_neg

                    if index == (len(self.beh) - 1):
                        self.pos_data[i][index] = train_v[i]
                    elif (uid, iid_pos) not in beh_dok:
                        if len(self.behaviors_data[index][uid].data) == 0:  # 如果用户根本没有该类型交互
                            self.pos_data[i][index] = -1
                        else:
                            t_array = self.behaviors_data[index][uid].toarray()
                            pos_index = np.where(t_array != 0)[1]
                            iid_pos = np.random.choice(pos_index, size=1, replace=True, p=None)[0]
                            self.pos_data[i][index] = iid_pos

                # not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]
                #
                #
                # user_id_list[index] = user[not_zero_index].long().cuda()
                # item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                # item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                # print(len(self.pos_data[index]))
                # print(type(self.pos_data[index]))
                # print(type(self.pos_data[index][0]))
                # print(self.pos_data[index][0].size)
                # print(len(self.pos_data[index][0]))
                #
                # assert 1==2
                # beh_item_set = self.pos_data[index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        # print(user.shape)
        # print(item_i.shape)
        # print(len(user))
        # print(len(item_i))
        # assert 1==2

        if self.is_training:
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i


class RecDataset_beh2(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):
        super(RecDataset_beh2, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh  # ['tip', 'neg', 'neutral', 'pos']
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0]  # user_num
        self.neg_data = [None] * self.length
        self.pos_data = [None] * self.length

    def ng_sample1(self):
        assert self.is_training, 'no need to sampling when testing'

        for i in range(self.length):
            self.neg_data[i] = [None] * len(self.beh)
            self.pos_data[i] = [None] * len(self.beh)

            # self.neg_data[i] = [None]
            # self.pos_data[i] = [None]

        for index in range(len(self.beh)):
            # if args.target == 'buy' and index != len(self.beh)-1:
            #     continue
            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()

            # train_u, train_v = self.[index].nonzero()
            # beh_dok = self.behaviors_data[index].todok()

            set_pos = np.array(list(set(train_v)))

            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

            for i in range(self.length):  #

                uid = self.data[i][0]
                iid_neg = self.neg_data[i][index] = self.neg_data_index[i]
                iid_pos = self.pos_data[i][index] = self.pos_data_index[i]

                if (uid, iid_neg) in beh_dok:
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)
                        self.neg_data[i][index] = iid_neg
                    self.neg_data[i][index] = iid_neg

                if index == (len(self.beh) - 1):
                    self.pos_data[i][index] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index][uid].data) == 0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index][uid].toarray()
                        pos_index = np.where(t_array != 0)[1]
                        iid_pos = np.random.choice(pos_index, size=1, replace=True, p=None)[0]
                        self.pos_data[i][index] = iid_pos

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        if args.target == 'buy':
            for i in range(self.length):
                self.neg_data[i] = [None] * 1
                self.pos_data[i] = [None] * 1
        else:
            for i in range(self.length):
                self.neg_data[i] = [None] * len(self.beh)
                self.pos_data[i] = [None] * len(self.beh)

        if args.target == 'buy':
            index = len(self.beh) - 1
            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()
            set_pos = np.array(list(set(train_v)))
            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

            for i in range(self.length):  #

                uid = self.data[i][0]
                iid_neg = self.neg_data[i][0] = self.neg_data_index[i]
                iid_pos = self.pos_data[i][0] = self.pos_data_index[i]

                if (uid, iid_neg) in beh_dok:  # 取neg sample
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)
                        self.neg_data[i][0] = iid_neg
                    self.neg_data[i][0] = iid_neg

                if index == (len(self.beh) - 1):
                    self.pos_data[i][0] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index][uid].data) == 0:
                        self.pos_data[i][0] = -1
                    else:
                        t_array = self.behaviors_data[index][uid].toarray()
                        pos_index = np.where(t_array != 0)[1]
                        iid_pos = np.random.choice(pos_index, size=1, replace=True, p=None)[0]
                        self.pos_data[i][0] = iid_pos


        else:
            for index in range(len(self.beh)):
                # if args.target == 'buy' and index != len(self.beh)-1:
                #     continue
                train_u, train_v = self.behaviors_data[index].nonzero()
                beh_dok = self.behaviors_data[index].todok()

                # train_u, train_v = self.[index].nonzero()
                # beh_dok = self.behaviors_data[index].todok()

                set_pos = np.array(list(set(train_v)))

                self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
                self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

                for i in range(self.length):  #

                    uid = self.data[i][0]
                    iid_neg = self.neg_data[i][index] = self.neg_data_index[i]
                    iid_pos = self.pos_data[i][index] = self.pos_data_index[i]

                    if (uid, iid_neg) in beh_dok:  # 取neg sample
                        while (uid, iid_neg) in beh_dok:
                            iid_neg = np.random.randint(low=0, high=self.num_item)
                            self.neg_data[i][index] = iid_neg
                        self.neg_data[i][index] = iid_neg

                    if index == (len(self.beh) - 1):
                        self.pos_data[i][index] = train_v[i]
                    elif (uid, iid_pos) not in beh_dok:
                        if len(self.behaviors_data[index][uid].data) == 0:
                            self.pos_data[i][index] = -1
                        else:
                            t_array = self.behaviors_data[index][uid].toarray()
                            pos_index = np.where(t_array != 0)[1]
                            iid_pos = np.random.choice(pos_index, size=1, replace=True, p=None)[0]
                            self.pos_data[i][index] = iid_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        if self.is_training:
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i


class RecDataset_beh1(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):
        super(RecDataset_beh1, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0]
        self.neg_data = [None] * self.length
        self.pos_data = [None] * self.length

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        for i in range(self.length):
            # self.neg_data[i] = [None] * len(self.beh)
            # self.pos_data[i] = [None] * len(self.beh)

            self.neg_data[i] = [None] * 1
            self.pos_data[i] = [None] * 1

        for index in range(len(self.beh)):
            if args.target == 'buy' and index != len(self.beh) - 1:
                continue
            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()

            # train_u, train_v = self.[index].nonzero()
            # beh_dok = self.behaviors_data[index].todok()

            set_pos = np.array(list(set(train_v)))

            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

            for i in range(self.length):  #

                uid = self.data[i][0]
                iid_neg = self.neg_data[i][0] = self.neg_data_index[i]
                iid_pos = self.pos_data[i][0] = self.pos_data_index[i]

                if (uid, iid_neg) in beh_dok:
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)
                        self.neg_data[i][0] = iid_neg
                    self.neg_data[i][0] = iid_neg

                if index == (len(self.beh) - 1):
                    self.pos_data[i][0] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[0][uid].data) == 0:
                        self.pos_data[i][0] = -1
                    else:
                        t_array = self.behaviors_data[0][uid].toarray()
                        pos_index = np.where(t_array != 0)[1]
                        iid_pos = np.random.choice(pos_index, size=1, replace=True, p=None)[0]
                        self.pos_data[i][0] = iid_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        if self.is_training:
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i


# import h5py
# temp = np.array([1,2,3,4,5])
# print(predir)
# with open(args.dataset+'/temp/temp_data.pkl','wb') as f:
#     pickle.dump(temp,f)

import os

