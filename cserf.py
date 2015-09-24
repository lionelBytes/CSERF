import numpy as np
from joblib import Parallel, delayed
from scipy.weave import inline

class FeatureCosts:
    def __init__(self, feature_costs, feature_depths, feature_names, feature_short_names, verbose_on = True):

        self.f_names = feature_names
        self.f_short_names = feature_short_names
        self.f_set_costs = feature_costs
        self.f_set_depths = feature_depths
        self.n_f_subsets = len(self.f_set_depths)
        self.f_purchased = [False] * self.n_f_subsets
        self.f_purchase_history = np.zeros((self.n_f_subsets, self.n_f_subsets ), dtype=bool) #creates a square array of 'False'
        self.f_running_costs = np.zeros(sum(self.f_set_depths))
        self.f_acquisition_points = np.array([np.NAN] * self.n_f_subsets)
        self.f_purchase_count = 0
        for idx in range(len(self.f_set_depths)):
            start_idx = np.sum(self.f_set_depths[0:idx])
            stop_idx = np.sum(self.f_set_depths[0:idx+1])
            self.f_running_costs[start_idx:stop_idx] = self.f_set_costs[idx]/self.f_set_depths[idx]
        self.verbose_on = verbose_on

    def update_feature_costs(self, chosen_feature, tree_no):

        subset_idx = np.where( chosen_feature < np.cumsum(self.f_set_depths))[0][0]
        self.f_purchased[subset_idx] = True #mark this feature subset as purchased
        self.f_purchase_history[self.f_purchase_count,:] = self.f_purchased
        self.f_purchase_count += 1
        start_idx = sum(self.f_set_depths[0:subset_idx])
        stop_idx = sum(self.f_set_depths[0:subset_idx+1])
        self.f_running_costs[start_idx:stop_idx] = 0.0
        subset_feature_no = int(chosen_feature - sum(self.f_set_depths[0:subset_idx]))

        self.f_acquisition_points[subset_idx] = tree_no

        if self.verbose_on:
            print 'Acquisition ' +str(self.f_purchase_count) +': '+ str(self.f_names[subset_idx]), \
                'feature number', subset_feature_no, 'was purchased along with whole', \
                self.f_short_names[subset_idx], 'subset (cost = '+str(self.f_set_costs[subset_idx])+ ',', \
                self.f_set_depths[subset_idx],'feature(s) in subset)'


class ForestParams:
    def __init__(self, cost_sensitivity, num_classes, decay=1, trees=105, depth=35, min_samp_cnt = 20, ):
        # settings.RF_NO_ACTIVE_VARS = '11';      % size of randomly selected subset of features to be tested at any given node (typically the sqrt of total no. of features)
        # settings.RF_GET_VAR_IMP = '1';          % calculate the variable importance of each feature during training (at cost of additional computation time)
        # settings.NO_OCCL_CLUSTERS = 1;          % the number of clusters to create out of features

        self.cost_sensitivity = cost_sensitivity
        self.num_tests = 150
        self.min_sample_cnt = min_samp_cnt
        self.max_depth = depth
        self.num_trees = trees
        self.bag_size = 0.5
        self.train_parallel = False
        self.num_classes = num_classes  # assumes that the classes are ordered from 0 to C
        self.verbose = False

class Node:

    def __init__(self, node_id, node_cnt, exs_at_node, impurity, probability):
        self.node_id = node_id  # id of absolute node
        self.node_cnt = node_cnt  # id not including nodes that didn't get made
        self.exs_at_node = exs_at_node
        self.impurity = impurity
        self.num_exs = float(exs_at_node.shape[0])
        self.is_leaf = True
        self.info_gain = 0.0

        # output
        self.probability = probability.copy()
        self.class_id = probability.argmax()

        # node test
        self.test_ind1 = 0
        self.test_thresh = 0.0

    def update_node(self, test_ind1, test_thresh, info_gain):
        self.test_ind1 = test_ind1
        self.test_thresh = test_thresh
        self.info_gain = info_gain
        self.is_leaf = False

    def create_child(self, test_res, impurity, prob, child_type, node_cnt):
        # save absolute location in dataset
        inds_local = np.where(test_res)[0]
        inds = self.exs_at_node[inds_local]

        if child_type == 'left':
            self.left_node = Node(2*self.node_id+1, node_cnt, inds, impurity, prob)
        elif child_type == 'right':
            self.right_node = Node(2*self.node_id+2, node_cnt, inds, impurity, prob)

    def test(self, X):
        return X[self.test_ind1] < self.test_thresh

    def get_compact_node(self):
        # used for fast forest
        if not self.is_leaf:
            node_array = np.zeros(4)
            # dims 0 and 1 are reserved for indexing children
            node_array[2] = self.test_ind1
            node_array[3] = self.test_thresh
        else:
            node_array = np.zeros(2+self.probability.shape[0])
            node_array[0] = -1  # indicates that its a leaf
            node_array[1] = self.node_cnt  # the id of the node
            node_array[2:] = self.probability.copy()
        return node_array


class Tree:

    def __init__(self, tree_id, tree_params, tree_costs):
        self.tree_id = tree_id
        self.tree_params = tree_params
        self.num_nodes = 0
        self.compact_tree = None  # used for fast testing forest and small memory footprint
        self.costs = tree_costs

    def build_tree(self, X, Y, node):
        #self.optimise_node on the line below returns true or false depending on whether there's been a successful split
        if (node.node_id < ((2.0**self.tree_params.max_depth)-1)) and (node.impurity > 0.0) \
                and (self.optimize_node(np.take(X, node.exs_at_node, 0), np.take(Y, node.exs_at_node), node)):
                self.num_nodes += 2

                if self.tree_params.verbose: '\nbuilding LEFT subtree'
                self.build_tree(X, Y, node.left_node)

                if self.tree_params.verbose:print '\nbuilding RIGHT subtree'
                self.build_tree(X, Y, node.right_node)

    def train(self, X, Y):
        # no bagging
        #exs_at_node = np.arange(Y.shape[0])

        # bagging
        exs_at_node = np.random.choice(Y.shape[0], int(Y.shape[0]*self.tree_params.bag_size), replace=False)
        exs_at_node.sort()

        # compute impurity
        prob, impurity = self.calc_impurity(np.take(Y, exs_at_node), np.ones((exs_at_node.shape[0], 1), dtype='bool'))

        # create root
        self.root = Node(0, 0, exs_at_node, impurity, prob[:, 0])
        self.num_nodes = 1

        # build tree
        self.build_tree(X, Y, self.root)

        # make compact version for fast testing
        self.compact_tree, _ = self.traverse_tree(self.root, np.zeros(0))

    def traverse_tree(self, node, compact_tree_in):
        node_loc = compact_tree_in.shape[0]
        compact_tree = np.hstack((compact_tree_in, node.get_compact_node()))

        # no this assumes that the index for the left and right child nodes are the first two
        if not node.is_leaf:
            compact_tree, compact_tree[node_loc] = self.traverse_tree(node.left_node, compact_tree)
            compact_tree, compact_tree[node_loc+1] = self.traverse_tree(node.right_node, compact_tree)

        return compact_tree, node_loc

    def test(self, X):
        op = np.zeros((X.shape[0], self.tree_params.num_classes))
        # check out apply() in tree.pyx in scikitlearn

        # single dim test
        for ex_id in range(X.shape[0]):
            node = self.root
            while not node.is_leaf:
                if X[ex_id, node.test_ind1] < node.test_thresh:
                    node = node.right_node
                else:
                    node = node.left_node
            op[ex_id, :] = node.probability
        return op

    def test_fast(self, X):
        op = np.zeros((X.shape[0], self.tree_params.num_classes))
        tree = self.compact_tree  # work around

        #in memory: for non leaf  node - 0 is lchild index, 1 is rchild, 2 is dim to test, 3 is threshold
        #in memory: for leaf node - 0 is leaf indicator -1, 1 is the node id, the rest is the probability for each class
        code = """
        int ex_id, node_loc, c_it;
        for (ex_id=0; ex_id<NX[0]; ex_id++) {
            node_loc = 0;
            while (tree[node_loc] != -1) {
                if (X2(ex_id, int(tree[node_loc+2]))  <  tree[node_loc+3]) {
                    node_loc = tree[node_loc+1];  // right node
                }
                else {
                    node_loc = tree[node_loc];  // left node
                }

            }

            for (c_it=0; c_it<Nop[1]; c_it++) {
                OP2(ex_id, c_it) = tree[node_loc + 2 + c_it];
            }
        }
        """
        inline(code, ['X', 'op', 'tree'])
        return op

    def get_leaf_ids(self, X):
        op = np.zeros((X.shape[0]))
        tree = self.compact_tree  # work around

        #in memory: for non leaf  node - 0 is lchild index, 1 is rchild, 2 is dim to test, 3 is threshold
        #in memory: for leaf node - 0 is leaf indicator -1, 1 is the node id, the rest is the probability for each class
        code = """
        int ex_id, node_loc;
        for (ex_id=0; ex_id<NX[0]; ex_id++) {
            node_loc = 0;
            while (tree[node_loc] != -1) {
                if (X2(ex_id, int(tree[node_loc+2]))  <  tree[node_loc+3]) {
                    node_loc = tree[node_loc+1];  // right node
                }
                else {
                    node_loc = tree[node_loc];  // left node
                }

            }

            op[ex_id] = tree[node_loc + 1];  // leaf id

        }
        """
        inline(code, ['X', 'op', 'tree'])
        return op

    def calc_impurity(self, y_local, test_res):

        prob = np.zeros((self.tree_params.num_classes, test_res.shape[1])) #probability of each class in each test

        # estimate probability
        # TODO could vectorize this with broadcasting
        for cc in range(self.tree_params.num_classes):
            node_test = test_res * (y_local[:, np.newaxis] == cc)
            prob[cc, :] = node_test.sum(axis=0)

        # normalize - make sure not to divide by zero
        prob[:, prob.sum(0) == 0] = 1.0
        prob = prob / prob.sum(0) #one Bernoulli distribution for each test split. We want a nice clean split...
        if self.tree_params.verbose: print '\nprob.shape', prob.shape

        # classification
        #impurity = -np.sum(prob*np.log2(prob))  # entropy
        impurity = 1.0-(prob**2).sum(0)  # gini
        if self.tree_params.verbose: print 'impurity.shape', impurity.shape

        return prob, impurity

    def node_split(self, x_local):
        # left node is false, right is true
        # single dim test
        if self.tree_params.verbose: print 'x_local.shape', x_local.shape

        # test_inds_1 stores the FEATURES across which each divide is tested
        test_inds_1 = np.sort(np.random.random_integers(0, x_local.shape[1]-1, self.tree_params.num_tests))
        x_local_expand = x_local.take(test_inds_1, 1)
        x_min = x_local_expand.min(0)
        x_max = x_local_expand.max(0)
        test_thresh = (x_max - x_min)*np.random.random_sample(self.tree_params.num_tests) + x_min
        # print 'test_inds_1', test_inds_1
        # print 'x_local_expand.shape', x_local_expand.shape
        # print 'x_min.shape', x_min.shape
        # print 'x_max.shape', x_max.shape
        #valid_var = (x_max != x_min)

        test_res = x_local_expand < test_thresh

        return test_res, test_inds_1, test_thresh

    def optimize_node(self, x_local, y_local, node):
        # TODO if num_tests is very large could loop over test_res in batches
        # TODO is the number of invalid splits is small it might be worth deleting the corresponding tests
        # %timeit rf.trees[0].optimize_node(X, Y, rf.trees[0].root)

        # perform split at node, producing outputs:
        # -test_res: nodeSamples x nTests. Binary array.
        # -tets_inds1: nTests. Integers. The chosen features (dimension) for each test.
        # -test_thresh: nTests. Floats. Threshold for each chosen dimension
        test_res, test_inds1, test_thresh = self.node_split(x_local)


        # count examples left and right
        num_exs_l = (~test_res).sum(axis=0).astype('float') #vector
        num_exs_r = x_local.shape[0] - num_exs_l  # i.e. num_exs_r = test_res.sum(axis=0).astype('float')
        valid_inds = (num_exs_l >= self.tree_params.min_sample_cnt) & (num_exs_r >= self.tree_params.min_sample_cnt)


        successful_split = False
        if valid_inds.sum() > 0: #if there was a single valid split
            # child node impurity
            prob_l, impurity_l = self.calc_impurity(y_local, ~test_res)
            prob_r, impurity_r = self.calc_impurity(y_local, test_res)

             # information gain - want the minimum
            num_exs_l_norm = num_exs_l/node.num_exs
            num_exs_r_norm = num_exs_r/node.num_exs
            #info_gain = - node.impurity + (num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l)
            info_gain = (num_exs_r_norm*impurity_r) + (num_exs_l_norm*impurity_l)

            # make sure we con only select from valid splits
            info_gain[~valid_inds] = info_gain.max() + 10e-10  # plus small constant


            split_costs = self.tree_params.cost_sensitivity * np.take(self.costs.f_running_costs, test_inds1)       # get the cost of the feature used in each test
            # print 'np.take(self.costs.f_short_names, test_inds1) ',\
            #     np.take(self.costs.f_short_names, test_inds1)
            # print 'split_costs ', split_costs

            # make sure we can only select from valid splits
            split_costs[~valid_inds] = split_costs.max() + 10e-10  # plus small constant
            J = info_gain + split_costs
            best_split = J.argmin()

            # print 'info_gain.argmax()', info_gain.argmax()
            # print 'split_costs.argmax()', split_costs.argmax()
            # print 'J[best_split]', J[best_split]

            if True:
            # if J[best_split] < 1.0:

                # if the info gain is acceptable split the node
                # TODO is this the best way of checking info gain?
                #if info_gain[best_split] > self.tree_params.min_info_gain:
                # create new child nodes and update node
                node.update_node(test_inds1[best_split], test_thresh[best_split], info_gain[best_split])
                node.create_child(~test_res[:, best_split], impurity_l[best_split], prob_l[:, best_split], 'left', self.num_nodes+1)
                node.create_child(test_res[:, best_split], impurity_r[best_split], prob_r[:, best_split], 'right', self.num_nodes+2)

                # if the feature had a non-zero cost, then update the running costs as the subset is now considered "purchased"
                if self.costs.f_running_costs[test_inds1[best_split]] != 0.0:
                    self.costs.update_feature_costs(test_inds1[best_split], self.tree_id)

                successful_split = True

        return successful_split


## Parallel training helper - used to train trees in parallel
def train_forest_helper(t_id, X, Y, params, seed):
    #print 'tree', t_id
    np.random.seed(seed)
    tree = Tree(t_id, params)
    tree.train(X, Y)
    return tree


class Forest:

    def __init__(self, params, feature_costs):
        self.params = params # Forest has param object
        self.trees = [] # Forest cosists of tree objects
        self.costs = feature_costs #Forest has feature costs object

    #def save(self, filename):
        # TODO make lightweight version for saving
        #with open(filename, 'wb') as fid:
        #    cPickle.dump(self, fid)

    def get_avg_test_cost(self, n_trees_input = None):
        if n_trees_input is not None:
            n_trees = n_trees_input
        else:
            n_trees = self.params.num_trees

        return np.sum([self.costs.f_acquisition_points < n_trees] * self.costs.f_set_costs)


    def train(self, X, Y, delete_old_trees):
        if delete_old_trees:
            self.trees = []

        # if self.params.train_parallel:
        #     # TODO Can I make this faster by sharing the data?
        #     #print 'Parallel training'
        #     # need to seed the random number generator for each process
        #     seeds = np.random.random_integers(0, 10e8, self.params.num_trees)
        #     self.trees.extend(Parallel(n_jobs=-1)(delayed(train_forest_helper)(t_id, X, Y, self.params, seeds[t_id])
        #                                      for t_id in range(self.params.num_trees)))
        # else:
        #print 'Standard training'

        ### ALWAYS TRAINS IN SERIAL

        for t_id in range(self.params.num_trees):
            if self.costs.verbose_on:
                print 'tree', t_id+1
            tree = Tree(t_id, self.params, self.costs)
            tree.train(X, Y)
            self.trees.append(tree)
            self.costs = tree.costs # update the latest feature acquisition costs from the tree to the forest
            # self.params.next_f_purchase_no = tree.next_f_purchase_no
            ### UPDATE FOREST PARAMS WITH NEW BASELINE SENSITIVITY AND RUNNING SENSITIVITY

        #print 'num trees ', len(self.trees)

    def test(self, X, n_trees_input = None):

        if n_trees_input is not None:
            n_trees = n_trees_input
        else:
            n_trees = self.params.num_trees

        op = np.zeros((X.shape[0], self.params.num_classes))
        for tt, tree in enumerate(self.trees[:n_trees]):
            # op_local = tree.test(X)  # regular python way - slow
            op_local = tree.test_fast(X)
            op += op_local
        op /= float(n_trees)
        return op

    def get_leaf_ids(self, X):
        op = np.zeros((X.shape[0], len(self.trees)), dtype=np.int64)
        for tt, tree in enumerate(self.trees):
            op[:, tt] = tree.get_leaf_ids(X)
        return op

    def delete_trees(self):
        del self.trees[:]
