import scipy.io as sio
import os, sys
import glob
import random_forest as rf
import cserf
import numpy as np
import pandas as pd
from pandas import read_csv
import time
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gc
gc.enable()
from matplotlib.pyplot import show, figure
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import colorsys
import stacked_bar_graph



class CSERF_Manager:

    def __init__(self, cserf_sensitivities, benchmark_RFs):

        self.sensitivities = cserf_sensitivities
        self.n_classifiers = len(self.sensitivities)
        print 'sensitivities of cserf classifiers', self.sensitivities
        self.algo_string = "cserf_"

        # following three arrays are typically unused in final version as val_forests and final forests are regrown from subsets found during the cserf growth process
        self.cserf_test_errors = np.zeros((runs, self.n_classifiers, n_cserf_samples ))
        self.cserf_AUCs = np.zeros((runs, self.n_classifiers, n_cserf_samples ))
        self.cserf_pred_f_costs = np.zeros((runs, self.n_classifiers, n_cserf_samples ))
        self.cserf_unused_importances = np.zeros((runs, self.n_classifiers, n_cserf_samples ))
        self.cserf_n_feats_used = [[] for i in range(len(dataset_obj.cost_strip_lbs))]

        # INITIALISE ARRAYS FOR STORING VALIDATION RFs CANDIDATES
        self.val_RFs_cand_ssets = [[[] for i in range(self.n_classifiers)] for j in range(runs)] #each entry will contain a list of unknown length
        self.val_RFs_cand_costs = [[[] for i in range(self.n_classifiers)] for j in range(runs)] #each entry will contain a list of unknown length
        self.val_RFs_cand_cols = [[[] for i in range(self.n_classifiers)] for j in range(runs)] #each entry will contain a list of unknown length
        self.val_RFs_cand_names = [[[] for i in range(self.n_classifiers)] for j in range(runs)] #each entry will contain a list of unknown length

        # FOR STORING THE SELECTED VALIDATION RANDOM FORESTS
        self.val_RFs_ssets = [[] for i in range(len(dataset_obj.cost_strip_lbs))]
        self.val_RFs_costs = [[] for i in range(len(dataset_obj.cost_strip_lbs))]
        self.val_RFs_cols = [[] for i in range(len(dataset_obj.cost_strip_lbs))]
        self.val_RFs_names = [[] for i in range(len(dataset_obj.cost_strip_lbs))]
        self.val_RFs_n_feats_used = [[] for i in range(len(dataset_obj.cost_strip_lbs))]


        # FOR STORING BENCHMARK RF PERFORMANCE ON VALIDATION AND TESTING SETS
        self.benchmark_RFs = benchmark_RFs
        self.benchmark_RF_Val_TE = np.zeros((final_RF_runs))
        self.benchmark_RF_Val_AUC = np.zeros((final_RF_runs))
        self.median_benchmark_Test_TE = np.zeros((nTestBatches))
        self.median_benchmark_Test_AUC = np.zeros((nTestBatches))


        #array for feature acquisition points, which record which tree a feature (subset) was purchased at - if at all
        self.f_acqn_points = np.empty((runs, len(dataset_obj.f_subset_costs), self.n_classifiers ))
        self.f_acqn_points[:] = np.NAN #records which tree each feature was purchased by during each run

        self.rf_subsets = np.zeros((runs, self.n_classifiers, n_cserf_samples, dataset_obj.n_f_subsets ), dtype=bool)
        self.rf_ss_names = [[[[] for i in range(n_cserf_samples)] for j in range(self.n_classifiers)] for k in range(runs)]
        self.rf_ss_cols = [[[[] for i in range(n_cserf_samples)] for j in range(self.n_classifiers)] for k in range(runs)]


    def grow_CSERFs(self, dataset_obj, x_train, y_train, x_val, y_val, runs, recombine_train_val_sets):

        self.cs_forests = []
        for idx, sensitivity in enumerate(self.sensitivities):
            self.rf_params = cserf.ForestParams(cost_sensitivity=sensitivity, num_classes=rf_num_classes,
                                                    trees=rf_max_num_trees, depth=rf_max_depth, min_samp_cnt = rf_min_sample_count) #  RF params --- note some params not implemented, such as features per node
            forest_costs = cserf.FeatureCosts(dataset_obj.f_subset_costs, dataset_obj.f_subset_depths,
                                                   dataset_obj.f_subset_names, dataset_obj.f_subset_short_names, verbose_on=verbose_cserfs)
            self.cs_forests.append( cserf.Forest(self.rf_params, forest_costs))

            print '\nRun', r+1,'of', runs,' - Training basic const sensitive forest', idx+1,'of', self.n_classifiers, 'with cost sensitivity: ', sensitivity
            # print 'x_train.shape', x_train.shape
            tic = time.time()
            self.cs_forests[idx].train(x_train, y_train, True)
            toc = time.time()
            self.f_acqn_points[r,:,idx] = self.cs_forests[idx].costs.f_acquisition_points
            print 'train time', toc-tic


    def predict_with_CSERFs(self, xVal, yVal):

        ### MODIFIED THIS FUNCTION ON 20TH AUG TO EXAMINE ONLY THE VALIDATION SET PIXELS / DATA

        for rf_idx, erf in enumerate(self.cs_forests): #one forest per sensitivity

            for samp_idx, n_trees in enumerate(cserf_sample_points):

                ## TESTING
                tic = time.time()
                y_pred_cserf_all_vals = erf.test(xVal, n_trees) # test the classifier using trees {0:n_trees}
                y_pred_cserf = y_pred_cserf_all_vals.argmax(1)
                y_pred_probs = 0.5 * (1 + y_pred_cserf_all_vals[:,1] - y_pred_cserf_all_vals[:,0])
                toc = time.time()
                if printTestTimes:
                    print 'CSERF with', n_trees,'trees test time:', toc-tic

                print 'xVal.shape: ', xVal.shape
                print 'y_pred_cserf_all_vals', y_pred_cserf_all_vals
                print 'y_pred_cserf', y_pred_cserf
                print 'yVal', yVal

                ### CALC AND SAVE TEST ERRORS
                self.cserf_test_errors[r, rf_idx, samp_idx] = 1 - (y_pred_cserf == yVal).mean()
                if rf_num_classes == 2:
                    self.cserf_AUCs[r, rf_idx, samp_idx] = calc_roc_auc(yVal, y_pred_probs, "CSERF classifier %d, using first %d trees, sensitivity=%.2f" % (rf_idx, n_trees ,self.sensitivities[rf_idx]))

                # SAVE EXPECTED TEST TIME FEATURE COST
                self.cserf_pred_f_costs[r, rf_idx, samp_idx] = erf.get_avg_test_cost(n_trees)
                print 'avg. test time cost of sensitivity', self.sensitivities[rf_idx],'classifier with with', n_trees,'trees:', self.cserf_pred_f_costs[r, rf_idx, samp_idx]

                # SAVE THE SUBSET INFORMATION
                self.rf_subsets[r, rf_idx, samp_idx, :] = erf.costs.f_acquisition_points < n_trees
                self.rf_ss_names[r][rf_idx][samp_idx] = [dataset_obj.f_subset_short_names[i]
                                                         for i, x in enumerate(erf.costs.f_acquisition_points < n_trees)
                                                         if x] # if the feature had already been purchased by the sample point, then add its name to the list
                cols_temp = [dataset_obj.f_ss_cols_dict[x] for x in self.rf_ss_names[r][rf_idx][samp_idx]]
                if len(cols_temp) > 0:
                    self.rf_ss_cols[r][rf_idx][samp_idx] = np.hstack((cols_temp[:]))
                self.cserf_unused_importances[r, rf_idx, samp_idx] = \
                    np.around(1 - np.sum(dataset_obj.avg_f_importances[self.rf_ss_cols[r][rf_idx][samp_idx]]), decimals=6)

        print 'self.cserf_unused_importances', self.cserf_unused_importances


    def analyse_CSERFs(self):
        print 'Forest test_times', self.cserf_pred_f_costs


        if rf_num_classes == 2:
            print 'CSERFs AUCs (sample points:)', self.cserf_AUCs
            boxplotter(self.cserf_AUCs, self.sensitivities, x_label = 'sensitivity',
                       title_string = 'CSERF AUCs on' +dataset_obj.fig_title_string+ 'val. set. (Trees=', title_vals = cserf_sample_points, legend_locn=4,
                       y_label = 'Lin Pen CSRF RF AUC', y_lim = [np.min(self.cserf_AUCs), np.max(self.cserf_AUCs)], full_range = True)
            scatter_plotter(self.cserf_AUCs, self.cserf_pred_f_costs, x_axis='Test-time Feature Cost', y_axis='Area under ROC',
                            legend_locn = 4, title_string = 'CSERF AUCs vs TTFC for' +dataset_obj.fig_title_string + 'val. set. (Trees=', technicolor_on = True,
                            title_vals = cserf_sample_points, y_lim = [np.min(self.cserf_AUCs), np.max(self.cserf_AUCs)])

        print 'CSERF test errors', self.cserf_test_errors
        print 'CSERF sensitivities', self.sensitivities
        print 'CSERF test error array shape', self.cserf_test_errors.shape
        print 'CSERF sensitivities array shape', self.sensitivities.shape

        ### BOX PLOTS OF TEST ERROR VS SENSITIVITY
        boxplotter(self.cserf_test_errors, self.sensitivities, x_label = 'sensitivity',
                   title_string = 'CSERF TEs on' +dataset_obj.fig_title_string+ 'val. set. (Trees=', title_vals = cserf_sample_points,legend_locn=1,
                   y_label = 'CSERF Test Error', y_lim = [np.min(self.cserf_test_errors), np.max(self.cserf_test_errors)], full_range=True)

        ### BOX PLOTS OF UNUSED F IMPORTANCES VS SENSITIVITY
        boxplotter(self.cserf_unused_importances, self.sensitivities, x_label = 'sensitivity',
                   title_string = 'CSERF IUF on' +dataset_obj.fig_title_string+ ' dataset (Trees=', title_vals = cserf_sample_points,legend_locn=1,
                   y_label = 'CSERF total importance of unused features', y_lim = [np.min(self.cserf_unused_importances), np.max(self.cserf_unused_importances)], full_range=True)

        ### BOX PLOTS OF TEST TIME VS SENSITIVITY
        boxplotter(self.cserf_pred_f_costs, self.sensitivities, x_label = 'sensitivity',
                   title_string = 'CSERF exp. TTFC on' +dataset_obj.fig_title_string+ 'dataset. (Trees=', title_vals = cserf_sample_points, legend_locn=1,
                   y_label = 'CSERF Expected Test Time Feature Cost', y_lim = [np.min(self.cserf_pred_f_costs), np.max(self.cserf_pred_f_costs)], full_range=True)

        ### SCATTER PLOT TEST TIME VS TEST ERROR
        scatter_plotter(self.cserf_test_errors, self.cserf_pred_f_costs, x_axis='Test-time Feature Cost', y_axis='Test Error on Validation Set', legend_locn = 1,
                        title_string = 'CSERF TEs vs TTFC on' +dataset_obj.fig_title_string + 'val. set. (Trees=', technicolor_on = True,
                        title_vals = cserf_sample_points, y_lim = [np.min(self.cserf_test_errors), np.max(self.cserf_test_errors)])

        ### SCATTER PLOT TEST TIME VS UNUSED F IMPORTANCE
        scatter_plotter(self.cserf_unused_importances, self.cserf_pred_f_costs, x_axis='Test-time Feature Cost',  y_axis='Importance of Unused Features', legend_locn = 1,
                        title_string = 'CSERF IUF vs TTFC,' +dataset_obj.fig_title_string + ' dataset (trees=', technicolor_on = True,
                        title_vals = cserf_sample_points, y_lim = [np.min(self.cserf_unused_importances), np.max(self.cserf_unused_importances)])

        ### SCATTER UNUSED F IMPORTANCE VS TEST ERROR
        scatter_plotter(self.cserf_test_errors, self.cserf_unused_importances, x_axis='Importance of Unused Features', y_axis='CSERF Test Errors on val. set', legend_locn = 2,
                        title_string = 'CSERF TEs vs IUF,' +dataset_obj.fig_title_string + ' dataset (trees=', technicolor_on = True,
                        title_vals = cserf_sample_points, y_lim = [np.min(self.cserf_test_errors), np.max(self.cserf_test_errors)])

        ### SET UP LOWER BOUND FOR STACKED BAR GRAPHS FOR ANALYSIS OF FEATURE ACQUISTION POINTS (which tree feature sets were bought in)
        self.acq_pt_buckets_lb, self.acq_pt_buckets_ub, self.bucket_labels = self.feat_acqn_choose_buckets() #get the lower and upper bound for each bucket
        self.n_buckets = len(self.acq_pt_buckets_lb) #the number of buckets
        self.f_purchase_groups = self.feat_acqn_group_points()

        print 'self.f_purchase_groups ', self.f_purchase_groups
        print 'self.f_purchase_groups.shape ', self.f_purchase_groups.shape

        if save_stacked_bar_graphs:
            self.feat_aqn_draw_stacked_bar_plots(self.f_purchase_groups)
        self.save_f_purchase_data(experiments_folder)
        self.save_f_purchase_bucketed_data(experiments_folder)
        self.save_testing_data(experiments_folder)


    def get_val_RFs_candidates_from_cserfs(self):


        # SAVE EXPECTED TEST TIME FEATURE COST
        for rf_idx, erf in enumerate(self.cs_forests): #one forest per sensitivity

            # print 'erf.costs.f_purchase_history[:erf.costs.f_purchase_count,:]', erf.costs.f_purchase_history[:erf.costs.f_purchase_count,:]
            self.val_RFs_cand_ssets[r][rf_idx] = erf.costs.f_purchase_history[:erf.costs.f_purchase_count,:]
            self.val_RFs_cand_costs[r][rf_idx] = [np.sum(self.val_RFs_cand_ssets[r][rf_idx][x,:] * erf.costs.f_set_costs)
                                                  for x in range(erf.costs.f_purchase_count)]
            self.val_RFs_cand_names[r][rf_idx] = [[] for x in range(erf.costs.f_purchase_count)]
            self.val_RFs_cand_cols[r][rf_idx] = np.zeros((erf.costs.f_purchase_count, dataset_obj.n_data_cols))

            for n_ss in range(erf.costs.f_purchase_count):

                # next line appends a name to the list for every 'True' in the row, where the n-th row stores which features were purchased
                # by the n-th acquisition. It's probably quite unintuitive, and is like this for legacy reasons - I want to re-write it 7
                # if I've got time!
                names_temp = [dataset_obj.f_subset_short_names[i] for i, x in enumerate(self.val_RFs_cand_ssets[r][rf_idx][n_ss,:]) if x]
                self.val_RFs_cand_names[r][rf_idx][n_ss]  = ','.join(names_temp)

                cols_temp = [dataset_obj.f_ss_cols_dict[x] for x in names_temp]
                cols_temp = np.hstack((cols_temp[:]))
                self.val_RFs_cand_cols[r][rf_idx][n_ss,:] = np.hstack((cols_temp, [None] * (dataset_obj.n_data_cols-cols_temp.shape[0])))
                # print str(n_ss)+'-th subset, cols_temp.shape:', cols_temp.shape

                # print 'self.val_RFs_cand_cols[r][rf_idx][n_ss] ', self.val_RFs_cand_cols[r][rf_idx][n_ss]

            print 'self.val_RFs_cand_costs[r][rf_idx]', self.val_RFs_cand_costs[r][rf_idx]
            print 'self.val_RFs_cand_names[r][rf_idx]', self.val_RFs_cand_names[r][rf_idx]
            # print 'self.val_RFs_cand_cols[r][rf_idx]', self.val_RFs_cand_cols[r][rf_idx]
            # print 'self.val_RFs_cand_ssets[r][rf_idx]', self.val_RFs_cand_ssets[r][rf_idx]


    def collate_val_RFs_cands(self):

        self.val_RFs_cand_ssets = np.vstack((self.val_RFs_cand_ssets[i][j][:] for i in range(runs) for j in range(self.n_classifiers)))
        self.val_RFs_cand_costs = np.hstack((self.val_RFs_cand_costs[i][j][:] for i in range(runs) for j in range(self.n_classifiers)))
        self.val_RFs_cand_names = np.hstack((self.val_RFs_cand_names[i][j][:] for i in range(runs) for j in range(self.n_classifiers)))
        self.val_RFs_cand_cols = np.vstack((self.val_RFs_cand_cols[i][j] for i in range(runs) for j in range(self.n_classifiers)))

        sort_inds = np.argsort(self.val_RFs_cand_costs)
        self.val_RFs_cand_ssets = self.val_RFs_cand_ssets[sort_inds,:]
        self.val_RFs_cand_costs = self.val_RFs_cand_costs[sort_inds]
        self.val_RFs_cand_names = self.val_RFs_cand_names[sort_inds]
        self.val_RFs_cand_cols = self.val_RFs_cand_cols[sort_inds,:]

        # self.val_RFs_ssets = [[] for i in range(len(dataset_obj.cost_strip_lbs))] self.val_RFs_costs = [[] for i in range(len(dataset_obj.cost_strip_lbs))] self.val_RFs_cols = [[] for i in range(len(dataset_obj.cost_strip_lbs))] self.val_RFs_names = [[] for i in range(len(dataset_obj.cost_strip_lbs))]

        for idx, lower_bound in enumerate(dataset_obj.cost_strip_lbs):

            strip_idxs = np.where((self.val_RFs_cand_costs >= lower_bound) & (self.val_RFs_cand_costs < dataset_obj.cost_strip_ubs[idx]))[0]

            # print 'self.val_RFs_cand_ssets[strip_idxs]  ', self.val_RFs_cand_ssets[strip_idxs]; print 'self.val_RFs_cand_cols[strip_idxs] ', self.val_RFs_cand_cols[strip_idxs]; print 'self.val_RFs_cand_costs[strip_idxs] ', self.val_RFs_cand_costs[strip_idxs]; print 'self.val_RFs_cand_names[strip_idxs] ', self.val_RFs_cand_names[strip_idxs]

            unique_sets, unique_set_idxs = np.unique(self.val_RFs_cand_names[strip_idxs], return_index=True)
            print 'unique_set_idxs', unique_set_idxs
            print 'strip_idxs', strip_idxs
            unique_set_idxs = strip_idxs[unique_set_idxs]
            n_unique_sets = unique_sets.shape[0]
            name_count = np.zeros((unique_sets.shape[0]))
            for set_idx, set_name in enumerate(unique_sets):
                name_count[set_idx] = np.sum(self.val_RFs_cand_names[strip_idxs] == set_name, dtype=float)

            probs = name_count/np.sum(name_count)

            print 'unique_sets ', unique_sets
            print 'probs', probs

            # print 'strip_idxs', strip_idxs print 'strip_idxs[0].shape', strip_idxs[0].shape print 'n_strip_samples', n_strip_samples print 'samples', samples;
            n_strip_samples = np.min((n_unique_sets, max_val_RFs_per_strip))
            print 'unique_set_idxs', unique_set_idxs
            if len(unique_sets) > 0: # if there was one or more unique set found
                sample_idxs = np.random.choice(unique_set_idxs, n_strip_samples, replace=False, p=probs)
                self.val_RFs_ssets[idx] = self.val_RFs_cand_ssets[sample_idxs,:]
                self.val_RFs_costs[idx] = self.val_RFs_cand_costs[sample_idxs]
                self.val_RFs_names[idx] = self.val_RFs_cand_names[sample_idxs]
                self.val_RFs_cols[idx] = self.val_RFs_cand_cols[sample_idxs,:]

                print 'self.val_RFs_ssets[idx].shape', self.val_RFs_ssets[idx].shape
                print 'self.val_RFs_costs[idx].shape', self.val_RFs_costs[idx].shape
                print 'self.val_RFs_cols[idx].shape', self.val_RFs_cols[idx].shape
                print 'self.val_RFs_names[idx].shape', self.val_RFs_names[idx].shape
            # samples = np.random.choice(strip_idxs[0], n_strip_samples, replace=False)

            print 'idx,', idx

        slice_not_empty = [len(self.val_RFs_ssets[i]) > 0 for i in range(len(dataset_obj.cost_strip_lbs))]

        for i in range(len(dataset_obj.cost_strip_lbs)):
            if len(self.val_RFs_ssets[i]) > 0 :
                print 'i', i, 'self.val_RFs_cols[i].shape', self.val_RFs_cols[i].shape
        self.val_RFs_ssets = np.vstack((self.val_RFs_ssets[i] for i in range(len(dataset_obj.cost_strip_lbs)) if slice_not_empty[i]))
        self.val_RFs_costs = np.hstack((self.val_RFs_costs[i] for i in range(len(dataset_obj.cost_strip_lbs)) if slice_not_empty[i]))
        self.val_RFs_names = np.hstack((self.val_RFs_names[i] for i in range(len(dataset_obj.cost_strip_lbs))if slice_not_empty[i]))
        self.val_RFs_cols = np.vstack((self.val_RFs_cols[i] for i in range(len(dataset_obj.cost_strip_lbs)) if slice_not_empty[i]))


        self.n_val_forests = self.val_RFs_names.shape[0]


        ### SORT BY COST
        sort_inds = np.argsort(self.val_RFs_costs)
        self.val_RFs_ssets = self.val_RFs_ssets[sort_inds,:]
        self.val_RFs_costs = self.val_RFs_costs[sort_inds]
        self.val_RFs_names = self.val_RFs_names[sort_inds]
        self.val_RFs_cols = self.val_RFs_cols[sort_inds,:]


        del self.cserf_AUCs
        del self.cserf_pred_f_costs
        del self.cserf_test_errors
        del self.cs_forests
        del self.val_RFs_cand_ssets
        del self.val_RFs_cand_costs
        del self.val_RFs_cand_names
        del self.val_RFs_cand_cols


    def train_test_val_RFs(self, xTrain, yTrain, xVal, yVal, experiments_folder, save_prefix = ''):

        #INITIALISE THE VALIDATION FORESTS
        if useOMAForest:
            self.val_RFs_OMA = [rf.Forest(rf_params_obj) for x in range(self.n_val_forests)]
            self.val_RFs_OMA_AUC = np.zeros((self.n_val_forests))
            self.val_RFs_OMA_TE = np.zeros((self.n_val_forests))

        if useSKLForest:
            self.val_RFs_SKL = [RandomForestClassifier(n_estimators=rf_max_num_trees, criterion=rf_criterion, max_depth=rf_max_depth,
                                                       min_samples_split=rf_min_sample_count, max_features=rf_no_active_vars, oob_score=False)
                                for x in range(self.n_val_forests)]
            self.val_RFs_SKL_AUC = np.zeros((self.n_val_forests))
            self.val_RFs_SKL_TE = np.zeros((self.n_val_forests))


        for rf_idx, rf_name in enumerate(self.val_RFs_names):

            print 'training forest', rf_idx+1, 'of', self.n_val_forests, ':', rf_name, ', with cost:', self.val_RFs_costs[rf_idx]

            subset_columns = self.val_RFs_cols[rf_idx, :]
            subset_columns = np.array(subset_columns[~np.isnan(subset_columns)], dtype=int)
            print 'subset_columns ', subset_columns
            x_tr = xTrain[:,subset_columns]
            x_vdtn = xVal[:,subset_columns]

            if useOMAForest:
                tic = time.time()
                self.val_RFs_OMA[rf_idx].train(x_tr, yTrain, True)
                print 'oma train time', time.time()-tic
                tic = time.time()
                y_pred_oma = self.val_RFs_OMA[rf_idx].test(x_vdtn).argmax(1)
                print 'oma validation testing time', time.time() - tic

                self.val_RFs_OMA_TE[rf_idx] = 1 - (y_pred_oma == yVal).mean()
                print 'OMA TE :', self.val_RFs_OMA_TE[rf_idx]
                if rf_num_classes == 2:
                    self.val_RFs_OMA_AUC[rf_idx] = calc_roc_auc(yVal, y_pred_oma, "OMA RF ")


            if useSKLForest:
                n_feat_per_node = np.sqrt(x_tr.shape[1]).astype(int)
                print 'n_feat_per_node', n_feat_per_node
                self.val_RFs_SKL[rf_idx].set_params(max_features = n_feat_per_node)
                tic = time.time()
                self.val_RFs_SKL[rf_idx].fit(x_tr, yTrain)
                print 'skl training time', time.time()-tic

                tic = time.time()
                y_pred_skl = self.val_RFs_SKL[rf_idx].predict(x_vdtn)
                y_pred_probs = self.val_RFs_SKL[rf_idx].predict_proba(x_vdtn)
                y_pred_probs = 0.5*(1+ y_pred_probs[:,1] - y_pred_probs[:,0])
                print 'skl validation testing time', time.time()-tic

                self.val_RFs_SKL_TE[rf_idx] = 1 - (y_pred_skl == yVal).mean()
                print 'SKL TE :', self.val_RFs_SKL_TE[rf_idx]

                # print 'y_pred_probs', y_pred_probs

                if rf_num_classes == 2:
                    self.val_RFs_SKL_AUC[rf_idx] = calc_roc_auc(yVal, y_pred_probs, "SKL RF ")


        ### GET THE BENCHMARK CLASSIFIER'S PERFORMANCE ON THE VALIDATION SET
        for r, forest in enumerate(self.benchmark_RFs):
            print 'Testing benchmark SKL RF', r, 'on the validation set'
            tic = time.time()
            y_pred_skl = forest.predict(xVal)
            y_pred_probs = forest.predict_proba(xVal)
            y_pred_probs = 0.5*(1+ y_pred_probs[:,1] - y_pred_probs[:,0])
            print 'Benchmark SKL RF testing time:', time.time() - tic
            self.benchmark_RF_Val_TE[r] = 1 - (y_pred_skl == yVal).mean()
            print 'benchmark SKL TE :', self.benchmark_RF_Val_TE[r]
            if rf_num_classes == 2:
                self.benchmark_RF_Val_AUC[r] = calc_roc_auc(yVal, y_pred_probs, "benchmark SKL RF ")

        self.median_benchmark_VAL_TE = np.median(np.array(self.benchmark_RF_Val_TE, dtype = float))
        print '\nMedian Benchmark Test Error on Val. Set:', self.median_benchmark_VAL_TE

        if rf_num_classes == 2:
            self.median_benchmark_VAL_AUC = np.median(np.array(self.benchmark_RF_Val_AUC, dtype = float))
            print '\nMedian Benchmark AUROC on Val. Set:', self.median_benchmark_VAL_AUC


        #ADD THE FULL FEATURE SET IN CASE ITS NOT IN THERE ALREADY
        if ','.join(dataset_obj.f_subset_short_names) not in self.val_RFs_names:
            self.val_RFs_ssets = np.vstack((self.val_RFs_ssets, np.ones((dataset_obj.n_f_subsets), dtype=bool)) )
            self.val_RFs_costs = np.hstack((self.val_RFs_costs, np.sum(dataset_obj.f_subset_costs)  ))
            self.val_RFs_names = np.hstack((self.val_RFs_names, 'SKL benchmark (all feature subsets)' ))
            self.val_RFs_cols = np.vstack((self.val_RFs_cols, range(x_train.shape[1]) ))
            self.val_RFs_SKL_TE = np.hstack((self.val_RFs_SKL_TE, self.median_benchmark_VAL_TE  ))

            if rf_num_classes == 2:
                self.val_RFs_SKL_AUC = np.hstack((self.val_RFs_SKL_AUC, self.median_benchmark_VAL_AUC  ))
        else:
            self.val_RFs_names[-1] = 'SKL benchmark (all feature subsets)'
            self.val_RFs_SKL_TE[-1] = self.median_benchmark_VAL_TE
            if rf_num_classes == 2:
                self.val_RFs_SKL_AUC[-1] = self.median_benchmark_VAL_AUC

        ### CALCULATE THE IMPORTANCES OF THE UNUSED FEATURES FOR EVERY VALIDATION RF
        self.val_RFs_unused_importances = np.zeros((self.val_RFs_costs.shape[0]))
        for i in range(len(self.val_RFs_unused_importances)):
            cols = self.val_RFs_cols[i,:]
            print 'cols[~np.isnan(cols)]', cols[~np.isnan(cols)]
            self.val_RFs_unused_importances[i] = np.around(1 - np.sum(dataset_obj.avg_f_importances[cols[~np.isnan(cols)].astype(int)]), decimals=6)


        print 'self.val_RFs_ssets', self.val_RFs_ssets
        print 'self.val_RFs_cost', self.val_RFs_costs
        print 'self.val_RFs_names ', self.val_RFs_names
        print 'self.val_RFs_cols ', self.val_RFs_cols
        print 'self.val_RFs_SKL_TE', self.val_RFs_SKL_TE
        print 'self.val_RFs_unused_importances ', self.val_RFs_unused_importances
        if rf_num_classes:
            print 'self.val_RFs_SKL_AUC', self.val_RFs_SKL_AUC


        ### BUILD THE DATAFRAME
        self.val_RFs_results = pd.DataFrame({'Subsets' : self.val_RFs_names,
                                        'Cost' : self.val_RFs_costs,
                                        'Unused Importances': self.val_RFs_unused_importances  })

        print 'self.val_RFs_results ', self.val_RFs_results

        self.val_RFs_results['SKL TE'] = self.val_RFs_SKL_TE
        scatter_plotter_from_df(self.val_RFs_results, 'Cost', 'SKL TE', "Val RF TE", baseline=self.median_benchmark_VAL_TE, legend_locn=1 , save_prefix=save_prefix , validation=True)
        if rf_num_classes == 2:
            self.val_RFs_results['SKL AUC'] = self.val_RFs_SKL_AUC
            scatter_plotter_from_df(self.val_RFs_results, 'Cost', 'SKL AUC', "Val RF AUC", baseline=self.median_benchmark_VAL_AUC, legend_locn=4 , save_prefix=save_prefix, validation=True )


        ### SAVE THE DATA FROM THE VALIDATION FORESTS
        directory,expt_path  = os.path.split(experiments_folder)
        directory_path = os.getcwd() + directory
        if not os.path.exists(directory_path + '/' + expt_path ):
            os.makedirs(directory_path + '/' + expt_path )
        savepath = directory_path + '/' + expt_path + '/' +save_prefix +'Val_RF_Results.csv'     # The final path to save to
        savepath = savepath.replace(" ", "_")
        self.val_RFs_results.to_csv(savepath, sep = ',', encoding='ascii')

        print 'self.val_RFs_results ', self.val_RFs_results


    def identify_final_forests(self, save_prefix=''):

        if useSKLForest:
            TEs = self.val_RFs_SKL_TE
        else:
            TEs = self.val_RFs_OMA_TE

        # Start the Pareto frontier with the lowest cost classifier
        p_front = [[self.val_RFs_costs[0], TEs[0] ]]
        print 'p_front', p_front
        final_RFs_idxs = [0]
        # Loop through the sorted list
        for pt_idx, pt_cost in enumerate(self.val_RFs_costs[1:]):
            print 'pt_cost', pt_cost
            print 'p_front[-1][1]', p_front[-1][1]
            print 'p_front[-1][0]', p_front[-1][0]
            print 'TEs[pt_idx]', TEs[pt_idx]
            if pt_cost == p_front[-1][0] and TEs[pt_idx+1] <= p_front[-1][1]:
                p_front.pop(-1)
                p_front.append([pt_cost, TEs[pt_idx+1]])
                final_RFs_idxs.pop(-1)
                final_RFs_idxs.append(pt_idx+1) #it's +1 because we started enumerating from the second point
            elif TEs[pt_idx+1] <= p_front[-1][1]:
                p_front.append([pt_cost, TEs[pt_idx+1]])
                final_RFs_idxs.append(pt_idx+1) #it's +1 because we started enumerating from the second point

        # Turn resulting pairs back into a list of Xs and Ys
        p_front = np.array(p_front)
        p_frontX = p_front[:,0]
        p_frontY = p_front[:,1]

        scatter_plotter_from_df(self.val_RFs_results, 'Cost', 'SKL TE', "Val RF TE(runs-" +str(runs)+ ")", pareto_on=True, pareto_data=p_front, baseline=self.median_benchmark_VAL_TE, save_prefix=save_prefix, validation=True )

        self.final_RFs_names = self.val_RFs_names[final_RFs_idxs]
        self.final_RFs_cols = self.val_RFs_cols[final_RFs_idxs,:]
        self.final_RFs_costs = self.val_RFs_costs[final_RFs_idxs]
        self.final_RFs_unused_importances = np.hstack((self.val_RFs_unused_importances[final_RFs_idxs], 0))

        #ADD THE FULL FEATURE SET IN CASE ITS NOT IN THERE ALREADY
        if 'SKL benchmark (all feature subsets)' not in self.final_RFs_names:
            print "Appending 'SKL benchmark (all feature subsets)' to the set of final classifiers (it didn't make the pareto front!)"
            self.final_RFs_costs = np.hstack((self.final_RFs_costs, self.val_RFs_costs[-1] ))
            self.final_RFs_names = np.hstack((self.final_RFs_names, self.val_RFs_names[-1] ))
            self.final_RFs_cols = np.vstack((self.final_RFs_cols, self.val_RFs_cols[-1,:] ))

        self.n_final_forests = len(self.final_RFs_costs)

        # DECLARE ARRAYS FOR STORING FINAL RESULTS NOW THAT FINAL NO. OF CLASSIFIERS IS KNOWN
        self.final_RFs_SKL_AUC = np.zeros((final_RF_runs, self.n_final_forests, nTestBatches))
        self.final_RFs_SKL_TE = np.zeros((final_RF_runs, self.n_final_forests, nTestBatches))


        print 'self.final_RFs_names', self.final_RFs_names
        print 'self.final_RFs_costs', self.final_RFs_costs

        print 'self.final_RFs_names.shape', self.final_RFs_names.shape
        print 'self.final_RFs_cols.shape', self.final_RFs_cols.shape
        print 'self.final_RFs_costs.shape', self.final_RFs_costs.shape

        del self.val_RFs_ssets
        del self.val_RFs_costs
        del self.val_RFs_names
        del self.val_RFs_cols
        del self.val_RFs_SKL
        del self.val_RFs_SKL_TE
        del self.val_RFs_SKL_AUC
        del self.val_RFs_unused_importances


    def train_final_forests(self, xTrain, yTrain, final_RF_runs):

        #INITIALISE THE FINAL FORESTS
        if useOMAForest:
            self.final_RFs_OMA = [[rf.Forest(rf_params_obj) for x in range(self.n_final_forests)] for r in range(final_RF_runs)]
            # self.final_RFs_OMA_AUC = np.zeros((final_RF_runs, self.n_final_forests))
            # self.final_RFs_OMA_TE = np.zeros((final_RF_runs, self.n_final_forests))
        if useSKLForest:
            self.final_RFs_SKL = [[RandomForestClassifier(n_estimators=rf_max_num_trees, criterion=rf_criterion, max_depth=rf_max_depth,
                                                            min_samples_split=rf_min_sample_count, max_features=rf_no_active_vars, oob_score=False)
                                                            for x in range(self.n_final_forests)] for r in range(final_RF_runs)]


        for set_idx, rf_set in enumerate(self.final_RFs_SKL):
            for rf_idx, forest in enumerate(rf_set):
                print 'run', set_idx, 'of',  final_RF_runs,', training forest', rf_idx+1, 'of', self.n_final_forests, ':', self.final_RFs_names[rf_idx], ', with cost:', self.final_RFs_costs[rf_idx]
                cols_temp = self.final_RFs_cols[rf_idx,:]
                # print 'cols_temp', cols_temp
                subset_columns = np.array(cols_temp[~np.isnan(cols_temp)], dtype = int)
                # print 'subset_columns', subset_columns
                x_tr_final = xTrain[:,subset_columns]

                if useOMAForest:
                    tic = time.time()
                    forest.train(x_tr_final, y_train, True)
                    print 'oma train time', time.time()-tic
                if useSKLForest:
                    forest.set_params(max_features = np.min((subset_columns.shape[0], rf_no_active_vars)))
                    tic = time.time()
                    forest.fit(x_tr_final, y_train)
                    print 'skl train time', time.time()-tic

        # if useSKLForest:
        #     self.final_RFs_SKL = [forest for forest_row in self.final_RFs_SKL for forest in forest_row]


    def test_final_forests(self, xTest, yTestIn, batch_idx, save_prefix = ''):


        if datasetNo in range(0,2):
            yTest = np.reshape(yTestIn, -1)
            print('\nRun ' +str(r)+ ' of ' + str(runs) + '. Testing scene: ' + str(dataset_obj.testSetScenes[batch_idx]))
            dataset_obj.show_ahmad_results(yTest, batch_idx)
            testImages = [np.zeros((yTest.shape[0])) for i in range(self.n_final_forests)]
        else:
            yTest = yTestIn


        for run, forest_set in enumerate(self.final_RFs_SKL):
            for rf_idx, forest in enumerate(forest_set):

                print 'run', run ,'testing forest', rf_idx+1, 'of', self.n_final_forests, ':', self.final_RFs_names[rf_idx], ', with cost:', self.final_RFs_costs[rf_idx]

                cols_temp = self.final_RFs_cols[rf_idx,:]
                subset_columns = np.array(cols_temp[~np.isnan(cols_temp)], dtype = int)
                x_te_final = xTest[:,subset_columns]

                y_pred_skl = forest.predict(x_te_final)

                if (datasetNo in range(0,2)) & (run == 0):
                    testImages[rf_idx] = forest.predict_proba(x_te_final)
                    ### save the predictions

                y_pred_probs = forest.predict_proba(x_te_final)
                y_pred_probs = 0.5*(1+ y_pred_probs[:,1] - y_pred_probs[:,0])

                self.final_RFs_SKL_TE[run, rf_idx, batch_idx] = 1 - (y_pred_skl == yTest).mean()
                print 'SKL TE :', self.final_RFs_SKL_TE[run, rf_idx]
                if rf_num_classes == 2:
                    self.final_RFs_SKL_AUC[run, rf_idx, batch_idx] = calc_roc_auc(yTest, y_pred_probs, "SKL RF ")

        if datasetNo in range(0,2):
            dataset_obj.show_predictions(testImages, batch_idx, dataset_obj.testSetScenes[batch_idx], save_prefix = save_prefix)

    def build_final_results_df(self, experiments_folder, save_prefix = ''):

        print 'self.final_RFs_names', self.final_RFs_names
        print 'self.final_RFs_costs', self.final_RFs_costs
        print 'self.final_RFs_unused_importances', self.final_RFs_unused_importances
        print 'self.final_RFs_names.shape', self.final_RFs_names.shape
        print 'self.final_RFs_costs.shape', self.final_RFs_costs.shape
        print 'self.final_RFs_unused_importances.shape', self.final_RFs_unused_importances.shape

        if self.final_RFs_unused_importances.shape[0] > self.final_RFs_costs.shape[0]:
            self.final_RFs_unused_importances = self.final_RFs_unused_importances[:-1]

        self.final_results_df = pd.DataFrame({'Subsets': self.final_RFs_names,
                                              'Costs': self.final_RFs_costs,
                                              'Unused Importances': self.final_RFs_unused_importances  })

        for batch_idx in range(nTestBatches):

            if datasetNo in range(2):
                suffix = ' B' + str(dataset_obj.testSetScenes[batch_idx])
            elif nTestBatches==1:
                suffix = ''
            else:
                suffix = ' B' + str(batch_idx)

            ## ADD OMA RF TEST ERRORS AND AUCS TO THE DATAFRAME
            if useOMAForest:
                self.final_results_df['OMA TE' + suffix] = np.median(self.final_RFs_OMA_TE[:, :, batch_idx], axis=0)
                if rf_num_classes == 2:
                    self.final_results_df['OMA AUC' + suffix] = np.median(self.final_RFs_OMA_AUC[:, :, batch_idx], axis=0)

            ## ADD SKL RF TEST ERRORS AND AUCS TO THE DATAFRAME
            if useSKLForest:
                self.final_results_df['SKL TE' + suffix] = np.median(self.final_RFs_SKL_TE[:, :, batch_idx], axis=0)
                if rf_num_classes == 2:
                    self.final_results_df['SKL AUC' + suffix] = np.median(self.final_RFs_SKL_AUC[:, :, batch_idx], axis=0)
            # for run in range(final_RF_runs):
            #
            #     suffix2 = ' R' + str(run)
            #
            #     if datasetNo in range(2):
            #         suffix = ' B' + str(dataset_obj.testSetScenes[batch_idx])
            #     elif nTestBatches==1:
            #         suffix = ''
            #     else:
            #         suffix = ' B' + str(batch_idx)
            #
            #     ## ADD OMA RF TEST ERRORS AND AUCS TO THE DATAFRAME
            #     if useOMAForest:
            #         self.final_results_df['OMA TE' + suffix + suffix2] = self.final_RFs_OMA_TE[run, :, batch_idx]
            #         if rf_num_classes == 2:
            #             self.final_results_df['OMA AUC' + suffix + suffix2] = self.final_RFs_OMA_AUC[run, :, batch_idx]
            #
            #     ## ADD SKL RF TEST ERRORS AND AUCS TO THE DATAFRAME
            #     if useSKLForest:
            #         self.final_results_df['SKL TE' + suffix + suffix2] = self.final_RFs_SKL_TE[run, :, batch_idx]
            #         if rf_num_classes == 2:
            #             self.final_results_df['SKL AUC' + suffix + suffix2] = self.final_RFs_SKL_AUC[run, :, batch_idx]


        ### --- SAVE TO CSV ---
        directory,expt_path  = os.path.split(experiments_folder)
        directory_path = os.getcwd() + directory
        if not os.path.exists(directory_path + '/' + expt_path ):
            os.makedirs(directory_path + '/' + expt_path )
        savepath = directory_path + '/' + expt_path + '/' + save_prefix + 'FinalForestResults.csv'     # The final path to save to
        savepath = savepath.replace(" ", "_")
        self.final_results_df.to_csv(savepath, sep = ',', encoding='ascii')


        print 'self.final_results_df ', self.final_results_df


    def analyse_final_forests(self, save_prefix=''):

        print '(np.swapaxes(self.final_RFs_SKL_TE,1,2))', (np.swapaxes(self.final_RFs_SKL_TE,1,2))
        print 'self.final_RFs_costs', self.final_RFs_costs

        if datasetNo in range(2):
            titles = ['Scene: ' + str(dataset_obj.testSetScenes[x]) for x in range(nTestBatches)]
        else:
            titles = ['' for x in range(nTestBatches)]

        # boxplotter(self.cserf_AUCs, self.sensitivities, x_label = 'sensitivity',  title_string = 'CSERF AUCs. nTrees: ', title_vals = cserf_sample_points,
        #            y_label = 'Lin Pen CSRF RF AUC', y_lim = [np.min(self.cserf_AUCs), np.max(self.cserf_AUCs)], full_range = True)
        boxplotter(self.final_RFs_SKL_TE, self.final_RFs_costs, x_label = 'Expected test-time feature acquisition cost',
                   title_string = 'Final RF Test Errors on '  + dataset_obj.fig_title_string  + ' Test Set',
                   title_vals = titles, x_ticks_in = dataset_obj.final_RF_xlabels, y_label = 'Final RFs Test Error', full_range = True,
                   y_lim = dataset_obj.plot_TE_ylim, baseline_on=True, save_prefix = save_prefix)

        ### SCATTER UNUSED F IMPORTANCE VS TEST ERROR
        if dataset_obj in range(2):
            tvals = dataset_obj.testSetScenes
        else:
            tvals = None

        ### SHOW HOW COST RELATES TO IMPORTANCE OF UNUSED FEATURES
        scatter_plotter_from_df(self.final_results_df, 'Costs', 'Unused Importances', "Importance of Unused Features", legend_locn=1, save_prefix=save_prefix, validation=True)

        if rf_num_classes == 2:
            print 'Forest AUCs', self.final_RFs_SKL_AUC
            boxplotter(self.final_RFs_SKL_AUC, self.final_RFs_costs, x_label = 'Expected test-time feature acquisition cost',
                   title_string = 'Final RF AUCs on '  + dataset_obj.fig_title_string  + ' Test Set',
                   title_vals = titles, x_ticks_in = dataset_obj.final_RF_xlabels, legend_locn = 4, y_label = 'Final RFs AUCs on ' + dataset_obj.fig_title_string,
                    full_range = True , baseline_on=True, save_prefix = save_prefix)

        # if rf_num_classes == 2:
        #     self.val_RFs_results['SKL AUC'] = self.val_RFs_SKL_AUC
        #     scatter_plotter_from_df(self.val_RFs_results, 'Cost', 'SKL AUC', "Val RF AUC (runs-" +str(runs)+ ")", baseline=self.median_benchmark_VAL_AUC, legend_locn=4 )


    def create_random_val_RFs(self):

        print 'creating', self.n_val_forests, 'random subsets for Val. RFs'
        # vector below used to space out the subsets' sizes in an even manner
        ss_lengths = np.floor(np.linspace(1, dataset_obj.n_f_subsets, self.n_val_forests)).astype(int)


        self.val_RFs_ssets = np.zeros((self.n_val_forests, dataset_obj.n_f_subsets), dtype=bool)
        self.val_RFs_costs = np.zeros((self.n_val_forests))
        self.val_RFs_names = [[] for x in range(self.n_val_forests)]
        self.val_RFs_cols = np.empty((self.n_val_forests, dataset_obj.n_data_cols))
        self.val_RFs_cols[:] = np.NAN

        for rf_idx, n_ss in enumerate(ss_lengths):

            subset_idxs = np.random.choice(dataset_obj.n_f_subsets, n_ss, replace=False)
            subset_idxs.sort()

            self.val_RFs_ssets[rf_idx, subset_idxs] = True #randomly choose n_ss subsets
            self.val_RFs_costs[rf_idx] = np.sum(self.val_RFs_ssets[rf_idx,:] * dataset_obj.f_subset_costs)
            self.val_RFs_names[rf_idx] = [dataset_obj.f_subset_short_names[ss] for ss in subset_idxs]

            cols_temp = [dataset_obj.f_ss_cols_dict[x] for x in self.val_RFs_names[rf_idx]]
            cols_temp = np.hstack((cols_temp[:]))
            self.val_RFs_cols[rf_idx, :cols_temp.shape[0]] = cols_temp

        if np.unique(dataset_obj.f_subset_costs).shape[0] > 1:
            #then we need to sort these randomly generated RFs by their costs!
            cost_ranks_idxs = np.argsort(self.val_RFs_costs)
            self.val_RFs_ssets =  self.val_RFs_ssets[cost_ranks_idxs,:]
            self.val_RFs_costs =  self.val_RFs_costs[cost_ranks_idxs]
            self.val_RFs_names =  [self.val_RFs_names[x] for x in cost_ranks_idxs]
            self.val_RFs_cols = self.val_RFs_cols[cost_ranks_idxs,:]


    def greedy_val_RFs(self):

        ss_lengths = np.floor(np.linspace(1, dataset_obj.n_f_subsets, self.n_val_forests)).astype(int)

        if np.unique(dataset_obj.f_subset_costs).shape[0] == 1:
            print 'all subset costs identical - will greedily choose best n features for each subset'

            if ss_lengths.shape[0] > dataset_obj.n_f_subsets: # we'd be getting duplicates. remove the dupes.
                ss_lengths = np.arange(dataset_obj.n_f_subsets).astype(int)
                self.n_val_forests = ss_lengths.shape[0]

            print 'creating', self.n_val_forests, 'subsets with the greedy algorithm'
            # vector below used to space out the subsets' sizes in an even manner

            self.val_RFs_ssets = np.zeros((self.n_val_forests, dataset_obj.n_f_subsets), dtype=bool)
            self.val_RFs_costs = np.zeros((self.n_val_forests))
            self.val_RFs_names = [[] for x in range(self.n_val_forests)]
            self.val_RFs_cols = np.empty((self.n_val_forests, dataset_obj.n_data_cols))
            self.val_RFs_cols[:] = np.NAN

            val_RF_subsets = [dataset_obj.f_imp_idxs[:n_ss+1] for n_ss in ss_lengths]
            print 'val_RF_subsets ', val_RF_subsets

            for rf_idx in range(len(val_RF_subsets)):
                subset_idxs = val_RF_subsets[rf_idx]
                subset_idxs.sort()
                self.val_RFs_ssets[rf_idx, subset_idxs] = True #randomly choose n_ss subsets
                self.val_RFs_costs[rf_idx] = np.sum(self.val_RFs_ssets[rf_idx,:] * dataset_obj.f_subset_costs)
                self.val_RFs_names[rf_idx] = [dataset_obj.f_subset_short_names[ss] for ss in subset_idxs]

                print 'self.val_RFs_ssets[rf_idx]', self.val_RFs_ssets[rf_idx]
                print 'self.val_RFs_costs[rf_idx]', self.val_RFs_costs[rf_idx]
                print 'self.val_RFs_names[rf_idx]', self.val_RFs_names[rf_idx]

                cols_temp = [dataset_obj.f_ss_cols_dict[x] for x in self.val_RFs_names[rf_idx]]
                cols_temp = np.hstack((cols_temp[:]))
                self.val_RFs_cols[rf_idx, :cols_temp.shape[0]] = cols_temp


        else:

            poss_val_RF_costs = np.cumsum(np.sort(dataset_obj.f_subset_costs))
            print 'Possible greedy val RF costs', poss_val_RF_costs

            self.val_RFs_costs = np.unique([np.max(poss_val_RF_costs[poss_val_RF_costs <= ub]) for idx, ub in enumerate(dataset_obj.cost_strip_ubs)])
            print 'self.val_RFs_costs', self.val_RFs_costs

            ### SETUP EMPTY ARRAYS
            self.n_val_forests = self.val_RFs_costs.shape[0]
            self.val_RFs_ssets = np.zeros((self.n_val_forests, dataset_obj.n_f_subsets), dtype=bool)
            self.val_RFs_names = [[] for x in range(self.n_val_forests)]
            self.val_RFs_cols = np.empty((self.n_val_forests, dataset_obj.n_data_cols))
            self.val_RFs_cols[:] = np.NAN


            val_RF_n_ss = [np.where(np.cumsum(np.sort(dataset_obj.f_subset_costs)) == self.val_RFs_costs[x])[0][0]+1 for x in range( self.n_val_forests )]
            print 'val_RF_n_ss', val_RF_n_ss

            temp_val_RF_ssets = [dataset_obj.f_greedy_idxs[:val_RF_n_ss[x]] for x in range(self.n_val_forests)]
            print 'temp_val_RF_ssets', temp_val_RF_ssets


            for rf_idx in range(self.n_val_forests):
                self.val_RFs_ssets[rf_idx, temp_val_RF_ssets[rf_idx]] = True
                self.val_RFs_names[rf_idx] = [dataset_obj.f_subset_short_names[ss] for ss in temp_val_RF_ssets[rf_idx]]
                cols_temp = [dataset_obj.f_ss_cols_dict[x] for x in self.val_RFs_names[rf_idx]]
                cols_temp = np.sort(np.hstack((cols_temp[:])))
                self.val_RFs_cols[rf_idx, :cols_temp.shape[0]] = cols_temp

                print 'self.val_RFs_ssets[rf_idx,:]', self.val_RFs_ssets[rf_idx,:]
                print 'self.val_RFs_costs[rf_idx]', self.val_RFs_costs[rf_idx]
                print 'self.val_RFs_cols[rf_idx,:]', self.val_RFs_cols[rf_idx,:]
                print 'self.val_RFs_names[rf_idx]', self.val_RFs_names[rf_idx]


            #
            # self.val_RFs_costs = []
            #
            # self.val_RFs_ssets = []
            # self.val_RFs_names = [[] for x in range(self.n_val_forests)]
            # self.val_RFs_cols = np.empty((self.n_val_forests, dataset_obj.n_data_cols))
            # self.val_RFs_cols[:] = np.NAN
            # self.val_RFs_ssets = np.zeros((self.n_val_forests, dataset_obj.n_f_subsets), dtype=bool)
            # self.val_RFs_costs = np.zeros((self.n_val_forests))
            # self.val_RFs_names = [[] for x in range(self.n_val_forests)]
            # self.val_RFs_cols = np.empty((self.n_val_forests, dataset_obj.n_data_cols))
            # self.val_RFs_cols[:] = np.NAN



            # dataset_obj.avg_fss_importances[x]









            print 'deciding Va.l RF subsets using Greedy algorithm'


    def feat_acqn_choose_buckets(self):

        ntree = self.rf_params.num_trees
        if ntree >= 20:
            bucket_lb = np.hstack(([0,1], np.array([0.04, 0.1, 0.25, 0.50, 0.75]) * (ntree-1)))
        # elif ntree >= 20:
        #     bucket_lb = np.hstack(([0,1], np.array([0.1, 0.25, 0.5, 0.75]) * (ntree-1)))
        elif ntree >= 10:
            bucket_lb = np.hstack(([0,1], np.array([0.25, 0.5, 0.75]) * (ntree-1)))
        elif ntree >= 2:
            bucket_lb = np.array([0,1])
        else:
            bucket_lb = np.array(0)

        # bucket_lb = np.array([0,1,5, 15, 30, 45, 60, 75, 90])
        # bucket_ub = np.array([0,4,14,29, 44, 59, 74, 89, 105])

        bucket_lb = np.round(bucket_lb).astype(int)
        bucket_ub = np.hstack((np.array(0), bucket_lb[2:]-1, ntree-1))
        n_buckets = bucket_lb.shape[0]

        bucket_labels = [ 'tree '+str(bucket_lb[x]+1) + ' to tree '+ str(bucket_ub[x]+1) for x in range(n_buckets)]
        bucket_labels[0] = 'first tree'

        print "'acquisition point bucket_lb' lower bound tree_ids:", bucket_lb

        return bucket_lb, bucket_ub, bucket_labels


    def feat_acqn_group_points(self):

        acquisitions_counts = np.zeros((self.n_buckets, dataset_obj.n_f_subsets, self.n_classifiers))
        #each iteration counts the number of times a given feature fell into a given bucket for a given sensitivity
        for idx, lower_bound in enumerate(self.acq_pt_buckets_lb): #fill up the array slice by slice
            upper_bound = self.acq_pt_buckets_ub[idx]
            acquisitions_counts[idx, :, :] = ((self.f_acqn_points>=lower_bound)&(self.f_acqn_points<=upper_bound)).sum(0)
        # count the number of runs which features were purchased in each bucket
        return acquisitions_counts.astype(float)/runs


    def feat_aqn_draw_stacked_bar_plots(self, bucketed_data):

        sp_per_fig = 6
        # nImages, spRows, spCols = 16, 4, 4
        spRows, spCols = get_subplot_row_cols(sp_per_fig)
        nFigs = np.ceil(float(self.f_acqn_points.shape[1]+1)/(sp_per_fig-1)).astype(int)
        # nImages, spRows, spCols = get_subplot_row_cols(self.f_acqn_points.shape[1])

        # We will plot one stacked bar plot for each feature. Each stacked bar is for one classifier. The components show the proportion of runs in each bucket
        HSV_tuples = [(x*1.0/self.n_buckets, 0.5, 0.5) for x in range(self.n_buckets)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        d_colors = []
        for tup in RGB_tuples:
            tup = tuple(i * 255 for i in tup)
            d_colors.append ('#%02x%02x%02x' % tup)


        for fig_idx in range(nFigs):

            sbg = stacked_bar_graph.StackedBarGrapher()

            fig = figure()
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')

            for idx in range(sp_per_fig):

                sp_id = idx+(sp_per_fig-1)*fig_idx
                if sp_id == dataset_obj.n_f_subsets:
                    break #all features plotted, quit the loop and save

                ax = fig.add_subplot(spRows+1, spCols, idx+1 + (np.floor(idx/3).astype(int)*3) )        # plt.scatter(x,y)
                d = bucketed_data[:,sp_id,:]
                sbg.stackedBarPlot(ax, d.transpose(), d_colors, seriesLabels=self.bucket_labels, edgeCols=['#000000'] * 7)
                ax.set_xlabel('classifier cost sensitivity (not to scale)')
                ax.set_ylabel('ratio (number of times acquired / n_runs)')
                if dataset_obj.f_subset_depths[sp_id] > 1:
                    ax.set_title('feat. subset: \'' + str(dataset_obj.f_subset_short_names[sp_id]) + '\' acquisitions. Cost='+str(dataset_obj.f_subset_costs[sp_id]) +
                                 '.\n Importance=%.5f (Rank %d of %d)' % (dataset_obj.avg_fss_importances[sp_id], dataset_obj.fss_ranks[sp_id]+1 , dataset_obj.n_f_subsets))
                else:
                    ax.set_title('Feature \'' + str(dataset_obj.f_subset_short_names[sp_id]) + '\' acquisitions. Cost='+str(dataset_obj.f_subset_costs[sp_id])+
                                 '.\n Importance=%.5f (Rank %d of %d)' % (dataset_obj.avg_f_importances[sp_id], dataset_obj.f_ranks[sp_id]+1 , dataset_obj.n_f_subsets))
                ax.set_ylim((0,1))
                ax.set_yticks(np.linspace(0.,1.,6))
                ax.set_yticklabels(np.linspace(0.,1.,6))
                ax.set_xticks((np.arange(0,self.n_classifiers)))
                plt.xticks(rotation=dataset_obj.xlabel_rotation)
                ax.set_xticklabels((self.sensitivities))
                ax.set_xlim((-0.5, self.n_classifiers+0.5))
                ax.tick_params(axis='both', which='major', labelsize=14)

            show(block=False)

            plt.legend()

            filestring = '/' + dataset_obj.img_name_string + str(rf_params_obj.num_trees) + 'T_FeatureSetPurchaseBuckets' + str(fig_idx+1) + 'of' + str(nFigs)
            filestring = filestring.replace(' ', '_')
            if save_data:
                save_fig(experiments_folder+filestring, 'svg')
                save_fig(experiments_folder+filestring, 'png')

            plt.close()



                # filestring2 = '/FeatureSetPurchasePoints'
                # if save_data:
                #     save_3d_array(self.f_acqn_points, os.getcwd() + experiments_folder + filestring1 + '.txt')
                #     save_3d_array(bucketed_data, os.getcwd() + experiments_folder + filestring2 + '.txt')


    def save_testing_data(self, experiments_folder, verbose = True):


        # make meshgrids for sensitivity, run and scene
        runs_vec = np.reshape(np.arange(0, runs), (runs, 1, 1))

        # if hasattr(dataset_obj, 'testSetScenes'):
        #     samp_pt_vec = np.array(dataset_obj.testSetScenes)
        # else:
        #     samp_pt_vec = np.arange(0,nTestBatches)
        samp_pt_vec = np.reshape(cserf_sample_points, (1,n_cserf_samples,1))

        sens_vec = self.sensitivities.reshape((1,1,self.sensitivities.shape[0]))

        runs_mesh_grid = np.tile(runs_vec, ([1, self.n_classifiers, n_cserf_samples]))
        samp_pt_mesh_grid = np.tile(samp_pt_vec, ([runs, self.n_classifiers, 1]))
        sens_mesh_grid = np.tile(sens_vec, ([runs, 1, n_cserf_samples]))

        print 'runs_mesh_grid.shape', runs_mesh_grid.shape
        print 'samp_pt_mesh_grid.shape', samp_pt_mesh_grid.shape
        print 'sens_mesh_grid.shape', sens_mesh_grid.shape
        print 'self.cserf_test_errors.shape', self.cserf_test_errors.shape
        print 'self.cserf_pred_f_costs.shape', self.cserf_pred_f_costs.shape


        # build pandas dataframe by vectorising all of the arrays
        self.df_testing = pd.DataFrame({ 'Sensitivity': np.reshape(sens_mesh_grid, -1),
                                         'Run': np.reshape(runs_mesh_grid, -1),
                                         'Sampled at tree': np.reshape(samp_pt_mesh_grid, -1),
                                         'Test Error': np.reshape(self.cserf_test_errors, -1),
                                         'Cost of Features': np.reshape(self.cserf_pred_f_costs, -1)
                                         })
        if rf_num_classes == 2:
            self.df_testing['AUC'] = np.reshape(self.cserf_AUCs, -1)


        ### --- SAVE TO CSV ---
        directory,expt_path  = os.path.split(experiments_folder)
        directory_path = os.getcwd() + directory
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        savepath = directory_path + '/' + expt_path + '/cserf_test_info.csv'     # The final path to save to
        savepath = savepath.replace(" ", "_")
        self.df_testing.to_csv(savepath, sep = ',', encoding='ascii')

        if verbose:     print("Saving basic CS-RF testing info to '%s'... Done" % savepath),


    def save_f_purchase_data(self, experiments_folder, verbose = True):

        # make meshgrids for sensitivity, run and scene
        runs_vec = np.reshape(np.arange(0, runs), (runs, 1, 1))

        f_short_names = np.array(dataset_obj.f_subset_short_names)
        feat_ss_vec = np.reshape(f_short_names, (1,f_short_names.shape[0],1))

        sens_vec = self.sensitivities.reshape((1,1,self.sensitivities.shape[0]))

        runs_mesh_grid = np.tile(runs_vec, ([1, dataset_obj.n_f_subsets, self.n_classifiers]))
        f_ss_mesh_grid = np.tile(feat_ss_vec, ([runs, 1, self.n_classifiers]))
        sens_mesh_grid = np.tile(sens_vec, ([runs, dataset_obj.n_f_subsets, 1]))

        # build pandas dataframe by vectorising all of the arrays
        self.df_training = pd.DataFrame({ 'Sensitivity': np.reshape(sens_mesh_grid, -1),
                                          'Run': np.reshape(runs_mesh_grid, -1),
                                          'Feature Subset': np.reshape(f_ss_mesh_grid, -1),
                                          'Purchased at Tree': np.reshape(self.f_acqn_points, -1)
                                          })

        ### --- SAVE TO CSV ---
        directory,expt_path  = os.path.split(experiments_folder)
        directory_path = os.getcwd() + directory
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        savepath = directory_path + '/' + expt_path + '/cserf_f_aqn_pt_info.csv'     # The final path to save to
        savepath = savepath.replace(" ", "_")
        self.df_training.to_csv(savepath, sep = ',', encoding='ascii')

        if verbose:     print("Saving basic CS-RF feature purchase info to '%s'..." % savepath),


    def save_f_purchase_bucketed_data(self, experiments_folder, verbose = True):

        # TODO : currently broken - if i want to use this then need to produce one column for each bucket in the dictionary. Is it necessary though?
        # make meshgrids for sensitivity, run and scene

        f_short_names = np.array(dataset_obj.f_subset_short_names)
        feat_ss_vec = np.reshape(f_short_names, (f_short_names.shape[0],1))

        sens_vec = self.sensitivities.reshape((1,self.sensitivities.shape[0]))

        f_ss_mesh_grid = np.tile(feat_ss_vec, ([ 1, self.n_classifiers]))
        sens_mesh_grid = np.tile(sens_vec, ([ dataset_obj.n_f_subsets, 1]))

        print 'self.f_purchase_groups', self.f_purchase_groups
        bucket_str = 'B'+str(0)+':'+str(self.acq_pt_buckets_lb[0])+'-'+str(self.acq_pt_buckets_ub[0])
        print 'self.f_purchase_groups[0,:,:].reshape(-1).shape', self.f_purchase_groups[0,:,:].reshape(-1).shape
        print 'self.f_purchase_groups[0,:,:].shape', self.f_purchase_groups[0,:,:].shape


        print 'self.f_purchase_groups', self.f_purchase_groups

        print 'np.reshape(sens_mesh_grid, -1).shape', np.reshape(sens_mesh_grid, -1).shape
        print 'np.reshape(f_ss_mesh_grid, -1).shape', np.reshape(f_ss_mesh_grid, -1).shape

        # print 'self.f_purchase_groups.astype(int).reshape(-1).shape', self.f_purchase_groups.astype(int).reshape(-1).shape


        # build pandas dataframe by vectorising all of the arrays
        self.df_bucket_info = pd.DataFrame({ 'Sensitivity': np.reshape(sens_mesh_grid, -1),
                                          'Feature Subset': np.reshape(f_ss_mesh_grid, -1),
                                          })

        for idx in range(self.n_buckets):
            bucket_str = 'Bucket:'+str(int(self.acq_pt_buckets_lb[idx]))+'-'+str(int(self.acq_pt_buckets_ub[idx]))
            self.df_bucket_info[bucket_str] = self.f_purchase_groups[idx,:,:].reshape(-1)



        ### --- SAVE TO CSV ---
        directory,expt_path  = os.path.split(experiments_folder)
        directory_path = os.getcwd() + directory
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        savepath = directory_path + '/' + expt_path + '/cserf_f_aqn_bucket_info.csv'      # The final path to save to
        savepath = savepath.replace(" ", "_")
        self.df_bucket_info.to_csv(savepath, sep = ',', encoding='ascii')

        if verbose:     print("Saving basic CS-RF feature purchase info to '%s'..." % savepath),


def make_dataset_object(datasetNo, use_fast_prototype_params, grouped_feature_subsets):

    if datasetNo in np.arange(0,2):    # OCCLUSIONS EXPERIMENT SETUP
        dataset_obj = OcclusionsDatasetManager(datasetNo, use_fast_prototype_params, grouped_feature_subsets)
        nTestBatches = dataset_obj.testSetScenes.shape[0]
    else:
        nTestBatches = 1
        if datasetNo == 2:    # MINIBOONE EXPERIMENT SETUP
            dataset_obj = MiniBooNE_DatasetManager(use_fast_prototype_params)
        elif datasetNo == 3:    # FOREST COVER EXPERIMENT SETUP
            dataset_obj = ForestCoverDatasetManager(use_fast_prototype_params, grouped_feature_subsets)
        elif datasetNo == 4:
            dataset_obj = CIFAR10DatasetManager(use_fast_prototype_params)
        elif datasetNo == 5:
            dataset_obj = ClimatePopCrashDataset()
        elif datasetNo == 6:
            dataset_obj = FertilityDataset()
        elif datasetNo == 7:
            dataset_obj = VehicleDataset()
        elif datasetNo == 8:
            dataset_obj = HeartSPECTF_Dataset()
        elif datasetNo == 9:
            dataset_obj = CIFAR10DatasetManager(use_fast_prototype_params)
        else: print "INVALID EXPERIMENT NUMBER"; sys.exit()


    ### ADD FEATURES COMMON TO EACH DATASET OBJECT
    column_inds = np.cumsum(dataset_obj.f_subset_depths)
    dataset_obj.f_ss_cols_dict = { dataset_obj.f_subset_short_names[x]
                                       : np.arange( column_inds[x] - dataset_obj.f_subset_depths[x], column_inds[x], dtype=int )
                                       for x in range(dataset_obj.n_f_subsets) }
    dataset_obj.f_ss_inds_dict = { dataset_obj.f_subset_short_names[x] : x for x in range(dataset_obj.n_f_subsets) }
    dataset_obj.n_cols = sum(dataset_obj.f_subset_depths)




    ### PRINT OUT DATASET INFORMATION TO CONSOLE
    rf_num_classes = len(np.unique(dataset_obj.y_train_full)) # this will change depending on the experiment!
    print 'rf_num_classes ', rf_num_classes
    print 'feature subset depths:', dataset_obj.f_subset_depths
    print 'feature subset costs:', dataset_obj.f_subset_costs
    print 'feature subset names:', dataset_obj.f_subset_names
    print 'feature subset shortened names:', dataset_obj.f_subset_short_names

    return dataset_obj, nTestBatches ,rf_num_classes


class OcclusionsDatasetManager:

    def __init__(self, datasetNo, prototypeParamsOn, showImageOption = 1, displayScene = 3):

        # TODO: Setup ungrouped features

        print 'Override settings on:', use_fast_prototype_params

        if datasetNo==0 :
            self.dataset_version = 'Lean' #either Lean or Full
            self.cost_strip_lbs = np.linspace(0.0, 4.0, 41)
            self.cost_strip_ubs = np.hstack((self.cost_strip_lbs[1:], 5))
            self.fig_title_string = ' Occls. Lean '

        else:
            self.dataset_version = 'Full'
            self.cost_strip_lbs = np.linspace(0.0, 48, 25)
            self.cost_strip_ubs = np.hstack((self.cost_strip_lbs[1:], 50))
            self.fig_title_string = ' Occls. Full '

        print 'RUNNING OCCLUSIONS DATASET, VERSION:', self.dataset_version

        # OCCLUSIONS TRAINING AND TEST SETTINGS - THESE SETTINGS ARE DESIGNED FOR LONG RUNS AND HIGH PERFORMANCE
        self.maxNoTrainScenes = 26 # Useful to set lower scene for trying stuff. Set to -1 to load all train scenes.
        self.maxNoTestScenes = 26 # Useful to set lower scene for trying stuff. Set to -1 to load all test scenes.
        self.loadLeanCfrPredictions = False #Load Ahmad's lean classifier outputs for results comparison?
        self.loadFullCfrPredictions = False #Load Ahmad's full classifier outputs for results comparison?

        self.folder_save_prefix = "Occlns_"+self.dataset_version+'_'
        self.img_name_string = "Occ" + self.dataset_version
        # self.cserf_sensitivities = self.make_sensitivities()

        # OCCLUSIONS IMAGE DISPLAY SETTINGS
        self.show_image_option = showImageOption # 0: do not show, 1: show posterior p(occlusion) only, 2: show training features and posterior p(occlusion)
        self.displayScene = displayScene # choose a scene whose features we want to display


        ### THE OVERRIDE PARAMETERS USED FOR QUICK TESTING OF THE OCCLUSIONS DATA
        if use_fast_prototype_params:
            self.maxNoTestScenes = 2 # Useful to set lower scene for trying stuff. Set to -1 to load all test scenes.
            self.maxNoTrainScenes = 2 # Useful to set lower scene for trying stuff. Set to -1 to load all train scenes.
            # self.cserf_sensitivities = np.array([0, 5, 20, 75])


        self.xlabels = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
        self.xlabel_rotation = 90
        # if self.dataset_version == 'Lean':
        #     self.reduced_sens_plot_range = 70
        #     self.reduced_sens_plot_AUC_ylim = [0.75, 1.0]
        #     self.reduced_sens_plot_TE_ylim = [0.0, 0.2]
        self.reduced_sens_plot_range = 70
        self.reduced_sens_plot_AUC_ylim = [0.75, 1.0]
        self.reduced_sens_plot_TE_ylim = [0.0, 0.2]
        self.plot_TE_ylim = [0.00, 0.20] # to match the Feng Nan et al. paper

        # else:
        #     self.reduced_sens_plot_range =


        ### LOAD FEATURE COST AND SET DEPTH INFORMATION
        self.f_subset_depths = np.loadtxt(self.dataset_version + 'TrainSetFeatDepths.txt') #load the feature subset depths for this version of the classifier
        self.n_data_cols = np.sum(self.f_subset_depths, dtype=int)
        self.f_subset_costs = np.loadtxt(self.dataset_version +'TrainSetFeatCompTimes.txt') #load the feature subset computation times for this version of the classifier
        self.f_subset_names = load_feature_subset_names(self.dataset_version + 'CfrFeatNames.txt') #function load the full feature subset names
        self.f_subset_short_names = np.genfromtxt(self.dataset_version +'CfrFeatAbbrevs.txt' , dtype = str) #short names can be all loaded with single line
        self.f_subset_short_names = self.f_subset_short_names.tolist()
        self.n_f_subsets = len(self.f_subset_depths)


        # self.cserf_sensitivities = np.array([0, 2, 5, 10, 15, 20, 30, 40, 50, 60, 65, 70, 75, 80])
        cserf_min_sens = 0.01 * (1/np.max(self.f_subset_costs))
        cserf_max_sens = 0.2 * (1/np.min(self.f_subset_costs))
        if not use_fast_prototype_params:
            n_cserfs = 9
        else:
            n_cserfs = 3
        self.cserf_sensitivities = np.around(np.hstack((0, np.linspace(cserf_min_sens, cserf_max_sens, n_cserfs)[1:])), decimals = 3)
        print 'Occlusions CSERF sensitivities: ', self.cserf_sensitivities

        self.final_RF_xlabels = np.linspace(0., 4., 9)


        self.allTrainGTs, self.allTrainFVs, trainSetScenes = self.load_data(self.dataset_version + "TrainSetFilesMatrixOnly.txt", stopAt = self.maxNoTrainScenes) #allTrainGTs, allTrainFVs: both are lists of numpy arrays. One numpy array for each scene in the set.
        print('TRAINING SET SCENES: ' + str(trainSetScenes))

        ### LOAD PREDICTIONS OUTPUT BY AHMAD'S CLASSIFIERS
        if self.loadLeanCfrPredictions:
            print('loading lean classifier predictions')
            leanCfrGrp2PredFolder = 'C:\Users\l\Desktop\OCCLUSIONS\occlusions_result\LEAN1-ed_pc_tg_av_lv_cs-max_rc_ra_fa_fn-GRP2'
            self.leanClassifierPreds = self.load_predictions(leanCfrGrp2PredFolder)
            self.show_predictions(self.leanClassifierPreds)
        if self.loadFullCfrPredictions:
            print('loading full classifier predictions')
            fullCfrGrp2PredFolder = 'C:\Users\l\Desktop\OCCLUSIONS\FullClassifierOP\FINAL-ed_pb_pb_pc_st_stm_tg_av_lv_cs-max_rc_ra_fc_fc_fa_fn_sp'
            self.fullClassifierPreds = self.load_predictions(fullCfrGrp2PredFolder)

        ### STACK ALL X AND y FROM SCENES TOGETHER
        self.y_train_full, self.x_train_full = self.stack_GT_and_FVs()

        ### LOAD THE TEST SET
        self.y_test, self.x_test, self.testSetScenes = self.load_data(self.dataset_version + "TestSetFilesMatrixOnly.txt", stopAt = self.maxNoTestScenes) #allTrainGTs, allTrainFVs: both are lists of numpy arrays. One numpy array for each scene in the set.
        print('TEST SET SCENES: ' + str(self.testSetScenes))


    def load_predictions(self,folderString):
        os.chdir(folderString)
        print(os.getcwd())
        preds = []
        for file in glob.glob("*_prediction.data"):
            print('loading prediction data from ' + file.rstrip()) # the rstrip method removes the \n tag
            preds.append(np.loadtxt(file))
        os.chdir(workingDir)
        return preds


    def stack_GT_and_FVs(self):

        # put all of the ground truth training arrays into vectors, then stack them into a single vector
        vectorisedTrainGTs = []
        for idx in range(len(self.allTrainGTs)): vectorisedTrainGTs.append(np.reshape(self.allTrainGTs[idx], -1)) # the np.reshape(-1) converts the ground truth into a vector
        allTrainGTVect = np.hstack(vectorisedTrainGTs)

        #stack all the features vectors from all training scenes, warning1: huge array! (a few GB), warning2: doesn't work if feature vectors different lengths
        allTrainFVsStacked = np.vstack(self.allTrainFVs)
        print 'dimensions of entire feature matrix for all pixels from all scenes: ' + str(allTrainFVsStacked.shape)
        return allTrainGTVect, allTrainFVsStacked


    def load_data(self, fileString, stopAt = -1): # loads the ground truth and feature vectors, given the list of .mat files containing them

        allGTs = []
        allFVs = []
        scenesInSet = []

        with open(fileString) as f:
            count = 0
            for filePath in f:
                struct = sio.loadmat(filePath.rstrip(), appendmat=False)
                if count%2==0:
                    print('loading ground truth data from ' + filePath.rstrip()) # the rstrip method removes the \n tag
                    allGTs.append(np.array(struct['gtMask']))
                    # following three lines record the scene number
                    _, filename = os.path.split(filePath)
                    scene =  filename.split('_',1) ; # returns a list with a single entry
                    scenesInSet.append(int(scene[0]))
                else:
                    print('loading feature vector data from ' + filePath.rstrip()) # the rstrip method removes the \n tag
                    allFVs.append(np.array(struct['features']))
                count+=1
                if count/2 == stopAt:
                    break
        return allGTs, allFVs, np.array(scenesInSet)


    def make_train_test_set(self):


        # training set selection: get all indices for both labels, then randomly select a subset to train on
        #### --- note that counterintuitively, pixels marked 0 *ARE* occluded, and pixels labelled with 1s are not!
        occludedIdx = np.asarray(np.where(self.y_train_full == 0)[0])
        occludedIdx = np.random.choice(occludedIdx, np.ceil(rf_maxSamplesPerLabel*1.5), replace = False)
        nonOccludedIdx = np.asarray(np.where(self.y_train_full == 1)[0])
        nonOccludedIdx = np.random.choice(nonOccludedIdx, np.ceil(rf_maxSamplesPerLabel*1.5), replace = False)
        trainOccludedIdx = occludedIdx[:rf_maxSamplesPerLabel]
        valOccludedIdx = occludedIdx[rf_maxSamplesPerLabel:]
        trainNonOccludedIdx = nonOccludedIdx[:rf_maxSamplesPerLabel]
        valNonOccludedIdx = nonOccludedIdx[rf_maxSamplesPerLabel:]

        ### TRAINING SET CREATION:
        xTrainOccluded = self.x_train_full[trainOccludedIdx, :]
        xTrainNonOccluded = self.x_train_full[trainNonOccludedIdx, :]
        xTrain = np.vstack((xTrainOccluded, xTrainNonOccluded))
        yTrain = np.hstack(  ( np.zeros((rf_maxSamplesPerLabel)), np.ones((rf_maxSamplesPerLabel)) )  )
        xValOccluded = self.x_train_full[valOccludedIdx, :]
        xValNonOccluded = self.x_train_full[valNonOccludedIdx, :]
        xVal = np.vstack((xValOccluded, xValNonOccluded))
        yVal = np.hstack(  ( np.zeros((np.ceil(rf_maxSamplesPerLabel*0.5))), np.ones((np.ceil(rf_maxSamplesPerLabel*0.5))) )  )


        # print 'xTrain, yTrain, xTrain.shape, yTrain.shape', xTrain, yTrain, xTrain.shape, yTrain.shape
        return xTrain, yTrain, xVal, yVal, self.x_test, self.y_test


    def show_ahmad_results(self, yTest, batch_idx):

        ### SHOW AHMAD'S CLASSIFIER'S RESULTS
        if dataset_obj.loadLeanCfrPredictions:
            calc_roc_auc(yTest, -dataset_obj.leanClassifierPreds[batch_idx], "Ahmad's Lean Classifier")
        if dataset_obj.loadFullCfrPredictions:
            calc_roc_auc(yTest, -dataset_obj.fullClassifierPreds[batch_idx], "Ahmad's Full Classifier")


    def show_features(self, x, y, experiments_folder, sceneNo = 10 , save_prefix = ''): #draws the ground along with along with all of the features in many subplots

        sp_per_fig = 6
        spRows, spCols = get_subplot_row_cols(sp_per_fig)
        nFigs = np.ceil(float(np.sum(self.f_subset_depths)+1)/sp_per_fig).astype(int)


        for fig_idx in range(nFigs):


            # mng = plt.get_current_fig_manager() # create fig manager for this window
            # mng.window.state('zoomed') #use fig manager to maximise the window
            #
            # for idx in range(images_per_fig):
            #     predImg = x[:,depth+sum(self.f_subset_depths[0,0:idx-1])]
            #     print(predImg.shape)
            #     predImg.resize((480,640))
            #     fig.add_subplot(spRows, spCols, idx+2)
            #     plt.imshow(predImg)
            #
            #     # if idx == len(featSubsetDepths)-1: show(block=True)
            #     # else: show(block=False)
            #
            # show(block=False)

            fig = figure()
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')

            for idx in range(sp_per_fig):

                ax = fig.add_subplot(spRows, spCols, idx+1 )        # plt.scatter(x,y)

                sp_id = idx+(sp_per_fig-1)*fig_idx
                print 'sp_id', sp_id
                if sp_id == self.n_data_cols+1:
                    break #all features plotted, quit the loop and save

                if sp_id == 0:
                    plt.imshow(y)
                    ax.set_title('Scene ' + str(sceneNo) + 'Ground Truth')

                else:
                    plt.imshow(np.reshape(x[:,sp_id-1], (480, 640)))
                    subset_idx = np.where( sp_id-1 < np.cumsum(self.f_subset_depths))[0][0]
                    subset_feature_no = int(sp_id - 1 - sum(self.f_subset_depths[0:subset_idx]))

                    ax.set_title('Scene ' + str(sceneNo) + ", Feature Subset  '" + self.f_subset_short_names[subset_idx] + "' feat. no. " + str(subset_feature_no))
                    # featImg = x[:,depth+sum(self.f_subset_depths[0,0:idx-1])]
            #     print(predImg.shape)
            #     predImg.resize((480,640))
            #     fig.add_subplot(spRows, spCols, idx+2)
                # ax = fig.add_subplot(spRows+1, spCols, idx+1 + (np.floor(idx/3).astype(int)*3) )        # plt.scatter(x,y)
                #
                # predImg = np.reshape(y_pre[idx][:,1], (480,640))
                # plt.imshow(predImg)

                ax.set_xlabel('x')
                ax.set_ylabel('y')

            show(block=False)


            filestring = '/Scene' + str(sceneNo) +'outputs_fig' + str(fig_idx)
            filestring = filestring.replace(' ', '_')
            if save_data:
                save_fig(experiments_folder+filestring, 'svg')
                save_fig(experiments_folder+filestring, 'png')

            plt.close()


    def show_predictions(self, y_pre, batch_idx, scene_no, save_prefix = ''): #draws the ground along with along with all of the features in many subplots


        sp_per_fig = 6
        spRows, spCols = get_subplot_row_cols(sp_per_fig)
        nFigs = np.ceil(float(cserf_mgr.final_RFs_names.shape[0])/sp_per_fig).astype(int)

        print y_pre
        print y_pre[0]
        print y_pre[0].shape
        # raw_input('roar')

        for fig_idx in range(nFigs):

            fig = figure()
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')

            for idx in range(sp_per_fig):

                sp_id = idx+(sp_per_fig-1)*fig_idx
                print 'sp_id', sp_id
                if sp_id >= cserf_mgr.final_RFs_names.shape[0]:
                    break #all features plotted, quit the loop and save

                ax = fig.add_subplot(spRows+1, spCols, idx+1 + (np.floor(idx/3).astype(int)*3) )        # plt.scatter(x,y)
                # ax = fig.add_subplot(spRows+1, spCols, idx )        # plt.scatter(x,y)

                predImg = np.reshape(y_pre[sp_id][:,1], (480,640))
                plt.imshow(predImg)

                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Scene ' + str(scene_no) + ',Classifier Cost' +  str(np.around(cserf_mgr.final_RFs_costs[sp_id], decimals=2)) + \
                             '\nSubsets: ' + str(cserf_mgr.final_RFs_names[sp_id]) + ', AUC:' + str(np.around(cserf_mgr.final_RFs_SKL_AUC[0, sp_id, batch_idx], decimals=3)))
                # ax.set_ylim((0,1))
                # ax.set_yticks(np.linspace(0.,1.,6))
                # ax.set_yticklabels(np.linspace(0.,1.,6))
                # ax.set_xticks((np.arange(0,self.n_classifiers)))
                # plt.xticks(rotation=dataset_obj.xlabel_rotation)
                # ax.set_xticklabels((self.sensitivities))
                # ax.set_xlim((-0.5, self.n_classifiers+0.5))
                # ax.tick_params(axis='both', which='major', labelsize=14)

            show(block=False)


            filestring = '/' + save_prefix + 'Scene' + str(scene_no) +'outputs_fig' + str(fig_idx)
            filestring = filestring.replace(' ', '_')
            if save_data:
                save_fig(experiments_folder+filestring, 'svg')
                save_fig(experiments_folder+filestring, 'png')

            plt.close()


class MiniBooNE_DatasetManager:


    def __init__(self, prototyping_params_on):

        self.folder_save_prefix = 'MiniBooNE'
        print('Loading MiniBooNE Dataset...')
        dataset_path = "C:\Users\l\Desktop\OCCLUSIONS\other_datasets\MiniBooNE"
        # self.x_train_full = np.genfromtxt(dataset_path + '/MiniBooNE_PID.txt', skip_header=1)
        # self.y_train_full = np.hstack((np.ones(36499),np.zeros(93565)))

        # LOAD DATA PROVIDED BY FENG NAN / BUDGET_RF GUYS
        self.x_train_full = np.array(read_csv(dataset_path + '/xtr.csv', header=None)).transpose()
        self.y_train_full = np.genfromtxt(dataset_path + '/ytr.csv' , delimiter=',', dtype = int)
        self.y_train_full[self.y_train_full==-1]=0

        self.x_val = np.array(read_csv(dataset_path + '/xtv.csv', header=None)).transpose()
        self.y_val = np.genfromtxt(dataset_path + '/ytv.csv', delimiter=',', dtype=int)
        self.y_val[self.y_val==-1]=0

        ### the testing function expects multiple batches of test data, stored in a list of np arrays
        self.x_test = [np.array(read_csv(dataset_path + '/xte.csv', header=None)).transpose()]
        self.y_test = [np.genfromtxt(dataset_path + '/yte.csv', delimiter=',', dtype=int)]
        self.y_test[0][self.y_test[0]==-1]=0

        print 'self.x_train_full', self.x_train_full, 'self.x_val', self.x_val, 'self.x_test', self.x_test
        print 'self.y_train_full', self.y_train_full, 'self.y_val', self.y_val, 'self.y_test', self.y_test

        print(' Done')
        # self.cserf_sensitivities = np.linspace(0, 0.9, 10)
        self.f_subset_depths, self.f_subset_costs, self.n_f_subsets = gen_generic_feat_costs(self.x_train_full.shape[1])  #load the computation times for each feature subset
        self.f_subset_names, self.f_subset_short_names = gen_generic_feat_names(self.x_train_full.shape[1])

        self.n_data_cols = np.sum(self.f_subset_depths)

        # self.cost_strip_lbs = np.linspace(0.5, 45.5, 10)
        # self.cost_strip_ubs = np.hstack((self.cost_strip_lbs[1:], 50.5))
        self.cost_strip_lbs = np.linspace(0.5, 49.5, 50)
        self.cost_strip_ubs = np.hstack((self.cost_strip_lbs[1:], 50.5))

        self.cserf_sensitivities = unary_cost_dset_sensitivities
        self.xlabels = unary_cost_dset_sens_xticks
        self.final_RF_xlabels = np.linspace(0.0,50.00,11)
        # self.cserf_sensitivities = np.array([0., 0.2, 0.4, 0.6, 0.64, 0.66, 0.68, 0.7, 0.72])
        # self.xlabels = np.array([0., 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8])
        self.xlabel_rotation = 90

        self.reduced_sens_plot_range = 0.65
        self.reduced_sens_plot_AUC_ylim = [0.75, 1.0]
        self.reduced_sens_plot_TE_ylim = [0.0, 0.15]
        self.fig_title_string = ' MiniBooNE '
        self.img_name_string = "MBE_"

        self.plot_TE_ylim = [0.06, 0.22] # to match the Feng Nan et al. paper


    def make_train_test_set(self):
        # MiniBooNE DATASET:
        # Data Set Characteristics:   Multivariate
        # Number of Instances: 130065
        # Area: Physical
        # Attribute Characteristics: Real
        # Number of Attributes: 50
        # Associated Tasks: Classification
        # There are 45523/19510/65031 examples in training/validation/test sets.
        # The signal events come first, followed by the background events. Each line, after the first line has the 50 particle ID variables for one event.

        # nPos, nNeg = 36499, 93565
        # nData = float(self.x_train_full.shape[0])
        # tr_val_split_idx = 2 * rf_maxSamplesPerLabel
        # trainSplit = min( ( 0.5, (19510 + tr_val_split_idx)/nData ) ) # 35% of the datapoints were used for training in the BudgetRF paper
        # # trainSplit = min((0.35, 2*rf_maxSamplesPerLabel/nData)) # 35% of the datapoints were used for training in the BudgetRF paper
        # testSplit = 0.5 # as used in the BudgetRF paper
        # print 'trainSplit', trainSplit
        # print 'testSplit', testSplit
        # xTrVal, x_test, yTrVal, y_test = train_test_split(self.x_train_full, self.y_train_full, train_size=trainSplit, test_size=testSplit)
        #
        # xTrain, yTrain, xVal, yVal = xTrVal[:np.min((45523, tr_val_split_idx)) , :], yTrVal[:np.min((45523, tr_val_split_idx))  ], \
        #                              xTrVal[np.min((45523, tr_val_split_idx))  :, :], yTrVal[np.min((45523, tr_val_split_idx))  :]


        if (2 * rf_maxSamplesPerLabel) < self.x_train_full.shape[0]:
            rand_inds = np.random.choice(self.x_train_full.shape[0], 2 * rf_maxSamplesPerLabel, replace=False)
            x_train = self.x_train_full[rand_inds, :]
            y_train = self.y_train_full[rand_inds]
        else:
            x_train = self.x_train_full
            y_train = self.y_train_full

        print 'xTrain', x_train , '\n', 'yTrain', y_train , '\n', 'x_test', self.x_test , '\n', 'y_test', self.y_test , '\n'
        print 'xTrain.shape', x_train.shape , '\n', 'yTrain.shape', y_train.shape , '\n', 'self.x_test.shape', self.x_test[0].shape , '\n', 'self.y_test.shape', self.y_test[0].shape

        return x_train, y_train, self.x_val, self.y_val, self.x_test, self.y_test


class ForestCoverDatasetManager:

    # FOREST COVER DATASET
    # Number of Attributes:	12 measures, but 54 columns of data
    # -- (10 quantitative variables, 4 binary wilderness areas and 40 binary soil type variables)
    # -- 7 types of forest cover

    # -- Classification performance
    # 	-- 70% Neural Network (backpropagation)
    # 	-- 58% Linear Discriminant Analysis
    # ACCORDING TO UCI DOCUMENTATION:
    # 	-- first 11,340 records used for training data subset
    # 	-- next 3,780 records used for validation data subset
    # 	-- last 565,892 records used for testing data subset
    # HOWEVER IN BUDGET RF PAPER THE FOLLOWING IS SAID:
    # "There are 36,603, 15,688, 58,101 examples in training/validation/test sets."
    # 15/08/24 : They've now provided us with their datasets :)

    def __init__(self, prototype_params_on, cat_vars_packaged_on):


        # self.cserf_sensitivities = np.array([0., .1, 0.2, .3, 0.4, 0.44, 0.46, 0.48, 0.5, 0.52])
        # self.xlabels = np.array([0., .1, 0.2, .3, 0.4, 0.44, 0.48, 0.52])
        self.cserf_sensitivities = unary_cost_dset_sensitivities
        self.xlabels = unary_cost_dset_sens_xticks
        self.xlabel_rotation = 90
        # self.cserf_sensitivities = np.linspace(0.,0.45,4)
        self.folder_save_prefix = 'ForestCov'
        self.fig_title_string = ' Forest Cover Type '

        self.cost_strip_lbs = np.linspace(0.5, 53.5, 54)
        self.cost_strip_ubs = np.hstack((self.cost_strip_lbs[1:], 54.5))


        if grouped_feature_subsets:     self.img_name_string = "CovType_Group"
        else:                           self.img_name_string = "CovType_Ungroup"

        print ('LOADING FOREST COVERTYPE DATASET...')
        dataset_path = "C:\Users\l\Desktop\OCCLUSIONS\other_datasets\covtype/"

        # dataset = np.array(read_csv(dataset_path + 'covtype.data'))
        # dataset = np.array(read_csv(dataset_path + 'covtype.data', nrows=36603+15688+58101))
        # x_full, y_full = dataset[:, 0:54], dataset[:, 54]
        #
        # random.seed(28021984) #
        # new_xy_inds = np.arange(0, x_full.shape[0])
        # np.random.shuffle(new_xy_inds)
        # x_full = x_full[new_xy_inds, :]
        # y_full = y_full[new_xy_inds]


        ### SPLIT THE DATASET INTO CONSTITUENTS AS DEFINED IN THE BUDGET RF PAPER
        # LOAD DATA PROVIDED BY FENG NAN / BUDGET_RF GUYS
        self.x_train_full = np.array(read_csv(dataset_path + '/xtr.csv', header=None)).transpose()
        self.y_train_full = np.genfromtxt(dataset_path + '/ytr.csv' , delimiter=',', dtype = int)

        self.x_val = np.array(read_csv(dataset_path + '/xtv.csv', header=None)).transpose()
        self.y_val = np.genfromtxt(dataset_path + '/ytv.csv', delimiter=',', dtype=int)

        ### the testing function expects multiple batches of test data, stored in a list of np arrays
        self.x_test = [np.array(read_csv(dataset_path + '/xte.csv', header=None)).transpose()]
        self.y_test = [np.genfromtxt(dataset_path + '/yte.csv', delimiter=',', dtype=int)]

        print(' Done')

        ### FEATURE SUBSETS SETUP
        self.f_subset_names = load_feature_subset_names(dataset_path + 'covtype_feature_names.txt') #function load the full feature subset names
        self.f_subset_short_names = np.genfromtxt(dataset_path + 'covtype_feature_short_names.txt' , dtype = str) #short names can be all loaded with single line

        self.reduced_sens_plot_range = 0.48
        self.reduced_sens_plot_AUC_ylim = [0.75, 1.0]
        self.reduced_sens_plot_TE_ylim = [0.0, 0.32]

        if cat_vars_packaged_on:
            ### TREAT THE CATEGORICAL "ONE HOT" VARIABLES AS A SINGLE SET WITH A SINGLE PRICE FOR THE SET
            self.f_subset_depths = np.array([1,1,1,1,1,1,1,1,1,1,4,40])

            self.n_f_subsets = len(self.f_subset_depths)
            self.f_subset_costs = self.f_subset_depths.copy() #load the computation times for each feature subset
        else:
            ### TREAT EVERY BINARY (ONE HOT) VARIABLE AS ITS OWN INDEPENDENT FEATURE WITH ITS OWN PRICE
            self.f_subset_depths, self.f_subset_costs, self.n_f_subsets = gen_generic_feat_costs(self.x_train_full.shape[1])  #load the computation times for each feature subset

            # ENUMERATE THE CATEGORICAL FEATURE NAMES
            temp1 = [self.f_subset_names[10]+ str(i) for i in np.arange(1,5)]
            temp2 = [self.f_subset_names[11]+ str(i) for i in np.arange(1,41)]
            self.f_subset_names = self.f_subset_names[0:10] + temp1 + temp2
            print 'self.f_subset_names ', self.f_subset_names

            temp1 = [self.f_subset_short_names[10]+ str(i) for i in np.arange(1,5)]
            temp2 = [self.f_subset_short_names[11]+ str(i) for i in np.arange(1,41)]
            self.f_subset_short_names = self.f_subset_short_names[0:10].tolist() + temp1 + temp2
            # featSubsetNames = load_feature_subset_names(dataset_path + 'covtype_feature_names.txt') #function load the full feature subset names
            # featSubsetShortNames = np.genfromtxt(dataset_path + 'covtype_feature_short_names.txt' , dtype = str) #short names can be all loaded with single line

        self.final_RF_xlabels = [0,5,10,15,20,25,30,35,40,45,50,54]

        self.n_data_cols = np.sum(self.f_subset_depths)
        self.plot_TE_ylim = [0.1, 0.35]

        y_test_pop = [np.sum(self.y_test[:] == x) for x in np.arange(1,8)]
        print 'y_test_pop', y_test_pop


    def make_train_test_set(self):

        if rf_maxSamplesPerLabel * rf_num_classes > self.x_train_full.shape[0]:
            #TODO: need option to actually bag the same number of each label...
            return self.x_train_full, self.y_train_full, self.x_val, self.y_val, self.x_test, self.y_test
        else:
            # Subsample the training set
            trainSplit = float(rf_num_classes*rf_maxSamplesPerLabel) / self.x_train_full.shape[0] # 35% of the datapoints were used for training in the BudgetRF paper
            print 'trainSplit', trainSplit
            xTrain, _, yTrain, _ = train_test_split(self.x_train_full, self.y_train_full, train_size=trainSplit)


        print 'xTrain.shape', xTrain.shape
        print 'yTrain.shape', yTrain.shape
        return xTrain, yTrain, self.x_val, self.y_val, self.x_test, self.y_test


class CIFAR10DatasetManager:


    def __init__(self, prototyping_params_on):

        self.folder_save_prefix = 'CIFAR10'
        print('Loading CIFAR-10 Dataset...')
        dataset_path = "C:\Users\l\Desktop\OCCLUSIONS\other_datasets\Cifar10"

        # LOAD DATA PROVIDED BY FENG NAN / BUDGET_RF GUYS
        self.x_train_full = np.array(read_csv(dataset_path + '/xtr.csv', header=None)).transpose()
        self.y_train_full = np.genfromtxt(dataset_path + '/ytr.csv' , delimiter=',', dtype = int)
        self.y_train_full[self.y_train_full==-1]=0

        # for some reason there are a lot of NaNs in column training data no. 3590, so remove this sample
        self.x_train_full = np.vstack((self.x_train_full[:3590,:], self.x_train_full[3591:,:]))
        self.y_train_full = np.hstack((self.y_train_full[:3590], self.y_train_full[3591:]))


        self.x_val = np.array(read_csv(dataset_path + '/xtv.csv', header=None)).transpose()
        self.y_val = np.genfromtxt(dataset_path + '/ytv.csv', delimiter=',', dtype=int)
        self.y_val[self.y_val==-1]=0

        ### the testing function expects multiple batches of test data, stored in a list of np arrays
        self.x_test = [np.array(read_csv(dataset_path + '/xte.csv', header=None)).transpose()]
        self.y_test = [np.genfromtxt(dataset_path + '/yte.csv', delimiter=',', dtype=int)]
        self.y_test[0][self.y_test[0]==-1]=0

        print(' Done')

        # self.cserf_sensitivities = np.linspace(0, 1, 6)
        self.f_subset_depths, self.f_subset_costs, self.n_f_subsets = gen_generic_feat_costs(self.x_train_full.shape[1])  #load the computation times for each feature subset
        self.f_subset_names, self.f_subset_short_names = gen_generic_feat_names(self.x_train_full.shape[1])
        self.n_data_cols = np.sum(self.f_subset_depths)
        self.cserf_sensitivities = unary_cost_dset_sensitivities
        self.xlabels = np.array([0., 0.4, 0.8, 1.2, 1.6, 2.0])


        self.cost_strip_lbs = np.linspace(0.5, 398.5, 200)
        self.cost_strip_ubs = np.hstack((self.cost_strip_lbs[1:], 400.5))
        # self.cost_strip_lbs = np.linspace(0.5, 398.5, 200)
        # self.cost_strip_ubs = np.hstack((self.cost_strip_lbs[1:], 400.5))

        # self.cserf_sensitivities = np.linspace(0,2,11)
        # self.cserf_sensitivities = np.array([0., 0.2, 0.4, 0.6, 0.64, 0.66, 0.68, 0.7, 0.72])
        # self.xlabels = np.array([0., 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8])
        self.xlabel_rotation = 90

        self.final_RF_xlabels = np.linspace(0,400,11)


        # self.reduced_sens_plot_range = 0.65
        # self.reduced_sens_plot_AUC_ylim = [0.75, 1.0]
        # self.reduced_sens_plot_TE_ylim = [0.0, 0.15]
        self.fig_title_string = ' CIFAR10 '
        self.img_name_string = "CIFAR10_"

        self.plot_TE_ylim = [0.3, 0.44] # to match the Feng Nan et al. paper

    def make_train_test_set(self):
        # CIFAR-10 DATASET:
        # There are 19,761, 8,468, 10,000 examples in training/validation/test sets.
        # The signal events come first, followed by the background events. Each line, after the first line has the 50 particle ID variables for one event.
        # The data are binarized by combining the first 5 classes into one class and the others into the second class.

        if (2 * rf_maxSamplesPerLabel) < self.x_train_full.shape[0]:
            rand_inds = np.random.choice(self.x_train_full.shape[0], 2 * rf_maxSamplesPerLabel, replace=False)
            x_train = self.x_train_full[rand_inds, :]
            y_train = self.y_train_full[rand_inds]
        else:
            x_train = self.x_train_full
            y_train = self.y_train_full

        print 'xTrain', x_train , '\n', 'yTrain', y_train , '\n', 'x_test', self.x_test , '\n', 'y_test', self.y_test , '\n'
        print 'xTrain.shape', x_train.shape , '\n', 'yTrain.shape', y_train.shape , '\n', 'self.x_test.shape', self.x_test[0].shape , '\n', 'self.y_test.shape', self.y_test[0].shape

        return x_train, y_train, self.x_val, self.y_val, self.x_test, self.y_test


class ClimatePopCrashDataset:

    def __init__(self):

        self.folder_save_prefix = 'ClimatePop'
        print('Loading Climate Pop Crash Dataset...')
        dataset_path = "C:\Users\l\Desktop\OCCLUSIONS\other_datasets"

        self.fig_title_string = ' Climate Crash '
        self.img_name_string = "ClimPop_"

        self.x_train_full, self.y_train_full, self.x_val, self.y_val, self.x_test, self.y_test = \
            load_vanilla_dataset(dataset_path + '\climatePopCrash_dataset.csv')

        print(' Done')
        print 'self.x_train_full', self.x_train_full, 'self.x_val', self.x_val, 'self.x_test', self.x_test
        print 'self.y_train_full', self.y_train_full, 'self.y_val', self.y_val, 'self.y_test', self.y_test

        self.cserf_sensitivities = np.linspace(0, 1, 6)
        self.f_subset_depths, self.f_subset_costs, self.n_f_subsets = gen_generic_feat_costs(self.x_train_full.shape[1])  #load the computation times for each feature subset
        self.f_subset_names, self.f_subset_short_names = gen_generic_feat_names(self.x_train_full.shape[1])
        self.n_data_cols = np.sum(self.f_subset_depths)

        self.n_cost_strips, self.cost_strip_lbs, self.cost_strip_ubs = gen_generic_cost_strips(self.n_f_subsets)

        self.cserf_sensitivities = np.linspace(0,2,11)
        self.xlabels = np.array([0., 0.4, 0.8, 1.2, 1.6, 2.0])
        self.xlabel_rotation = 90

        self.plot_TE_ylim = None

    def make_train_test_set(self):

        x_train, y_train = generic_make_train_set(self)
        return x_train, y_train, self.x_val, self.y_val, self.x_test, self.y_test


class FertilityDataset:

    def __init__(self):

        self.folder_save_prefix = 'Fertility'
        print('Loading Fertility Dataset...')
        dataset_path = "C:\Users\l\Desktop\OCCLUSIONS\other_datasets"

        self.fig_title_string = ' Fertility '
        self.img_name_string = "Fertility_"

        self.x_train_full, self.y_train_full, self.x_val, self.y_val, self.x_test, self.y_test = \
            load_vanilla_dataset(dataset_path + '/fertility_dataset.csv')

        print(' Done')
        print 'self.x_train_full', self.x_train_full, 'self.x_val', self.x_val, 'self.x_test', self.x_test
        print 'self.y_train_full', self.y_train_full, 'self.y_val', self.y_val, 'self.y_test', self.y_test

        self.cserf_sensitivities = np.linspace(0, 1, 6)
        self.f_subset_depths, self.f_subset_costs, self.n_f_subsets = gen_generic_feat_costs(self.x_train_full.shape[1])  #load the computation times for each feature subset
        self.f_subset_names, self.f_subset_short_names = gen_generic_feat_names(self.x_train_full.shape[1])
        self.n_data_cols = np.sum(self.f_subset_depths)

        self.n_cost_strips, self.cost_strip_lbs, self.cost_strip_ubs = gen_generic_cost_strips(self.n_f_subsets)

        self.cserf_sensitivities = np.linspace(0,2,11)
        self.xlabels = np.array([0., 0.4, 0.8, 1.2, 1.6, 2.0])
        self.xlabel_rotation = 90

    def make_train_test_set(self):

        x_train, y_train = generic_make_train_set(self)
        return x_train, y_train, self.x_val, self.y_val, self.x_test, self.y_test


class VehicleDataset:

    def __init__(self):

        self.folder_save_prefix = 'Vehicle'
        print('Loading Vehicle Dataset...')
        dataset_path = "C:\Users\l\Desktop\OCCLUSIONS\other_datasets"

        self.fig_title_string = ' Vehicle '
        self.img_name_string = "Vehicle_"

        self.x_train_full, self.y_train_full, self.x_val, self.y_val, self.x_test, self.y_test = \
            load_vanilla_dataset(dataset_path + '/vehicle_dataset.csv')

        print(' Done')
        print 'self.x_train_full', self.x_train_full, 'self.x_val', self.x_val, 'self.x_test', self.x_test
        print 'self.y_train_full', self.y_train_full, 'self.y_val', self.y_val, 'self.y_test', self.y_test

        self.cserf_sensitivities = np.linspace(0, 1, 6)
        self.f_subset_depths, self.f_subset_costs, self.n_f_subsets = gen_generic_feat_costs(self.x_train_full.shape[1])  #load the computation times for each feature subset
        self.f_subset_names, self.f_subset_short_names = gen_generic_feat_names(self.x_train_full.shape[1])
        self.n_data_cols = np.sum(self.f_subset_depths)

        self.n_cost_strips, self.cost_strip_lbs, self.cost_strip_ubs = gen_generic_cost_strips(self.n_f_subsets)

        self.cserf_sensitivities = np.linspace(0,2,11)
        self.xlabels = np.array([0., 0.4, 0.8, 1.2, 1.6, 2.0])
        self.xlabel_rotation = 90

        self.plot_TE_ylim = None

    def make_train_test_set(self):

        x_train, y_train = generic_make_train_set(self)
        return x_train, y_train, self.x_val, self.y_val, self.x_test, self.y_test


class HeartSPECTF_Dataset:

    def __init__(self):

        self.folder_save_prefix = 'HeartSPECTF'
        print('Loading HeartSPECTF Dataset...')
        dataset_path = "C:\Users\l\Desktop\OCCLUSIONS\other_datasets"

        self.fig_title_string = ' heartSPECTF '
        self.img_name_string = "heartSPECTF"

        self.x_train_full, self.y_train_full, self.x_val, self.y_val, self.x_test, self.y_test = \
            load_vanilla_dataset(dataset_path + '/heartSPECTF_dataset.csv')

        print(' Done')
        print 'self.x_train_full', self.x_train_full, 'self.x_val', self.x_val, 'self.x_test', self.x_test
        print 'self.y_train_full', self.y_train_full, 'self.y_val', self.y_val, 'self.y_test', self.y_test

        self.cserf_sensitivities = np.linspace(0, 1, 6)
        self.f_subset_depths, self.f_subset_costs, self.n_f_subsets = gen_generic_feat_costs(self.x_train_full.shape[1])  #load the computation times for each feature subset
        self.f_subset_names, self.f_subset_short_names = gen_generic_feat_names(self.x_train_full.shape[1])
        self.n_data_cols = np.sum(self.f_subset_depths)

        self.n_cost_strips, self.cost_strip_lbs, self.cost_strip_ubs = gen_generic_cost_strips(self.n_f_subsets)

        self.cserf_sensitivities = np.linspace(0,2,11)
        self.xlabels = np.array([0., 0.4, 0.8, 1.2, 1.6, 2.0])
        self.xlabel_rotation = 90

        self.plot_TE_ylim = None

    def make_train_test_set(self):

        x_train, y_train = generic_make_train_set(self)
        return x_train, y_train, self.x_val, self.y_val, self.x_test, self.y_test


def get_avg_f_importances(benchmark_RFs, experiments_folder):

    ### GET FEATURE IMPORTANCES AND STANDARD DEVIATIONS FROM THE BENCHMARK FORESTS
    feature_importances = np.vstack((forest.feature_importances_ for forest in benchmark_RFs))
    print 'feature_importances ', feature_importances
    stds = [np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0) for forest in benchmark_RFs]
    print 'stds', stds
    dataset_obj.avg_f_importances = np.mean(feature_importances, axis=0)
    dataset_obj.avg_f_imp_stds = np.mean(stds, axis=0)

    dataset_obj.f_imp_idxs  = np.argsort(dataset_obj.avg_f_importances)[::-1]


    ### group the feature subset importances
    if grouped_feature_subsets:
        # group the feature importances for x in dataset_obj.f_subset_short_names: # print 'x', x, 'dataset_obj.f_ss_cols_dict[x]', dataset_obj.f_ss_cols_dict[x] print 'dataset_obj.avg_f_importances[dataset_obj.f_ss_cols_dict[x]]', dataset_obj.avg_f_importances[dataset_obj.f_ss_cols_dict[x]]
        dataset_obj.avg_fss_importances = np.array([np.sum(dataset_obj.avg_f_importances[dataset_obj.f_ss_cols_dict[x]])
                                           for x in dataset_obj.f_subset_short_names])
        for x in range(dataset_obj.n_f_subsets):
            print dataset_obj.f_subset_short_names[x],' subset importance:',  dataset_obj.avg_fss_importances[x]
        dataset_obj.f_ss_imp_idxs  = np.argsort(dataset_obj.avg_fss_importances)[::-1]

    print 'dataset_obj.f_imp_idxs  ', dataset_obj.f_imp_idxs
    print 'feature ranking:'
    for f in range(dataset_obj.n_data_cols):
        print("%d. feature %d (%f)" % (f + 1, dataset_obj.f_imp_idxs[f], dataset_obj.avg_f_importances[dataset_obj.f_imp_idxs[f]]))
    dataset_obj.f_ranks = np.argsort(dataset_obj.f_imp_idxs)
    dataset_obj.fss_ranks = np.argsort(dataset_obj.f_ss_imp_idxs)

    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(dataset_obj.fig_title_string + "Feature importances (sorted descending)", fontsize = 32)
    ax.set_ylabel('Feature importance', fontsize = 28)


    if not grouped_feature_subsets:
        series = dataset_obj.avg_f_importances[dataset_obj.f_imp_idxs]
        y_err = dataset_obj.avg_f_imp_stds[dataset_obj.f_imp_idxs]
    else:
        series = dataset_obj.avg_fss_importances[dataset_obj.f_ss_imp_idxs]
        y_err = None

    if dataset_obj.n_f_subsets < 150:
        plt.bar(range(dataset_obj.n_f_subsets), series, color="r", yerr=y_err, align="center")
        if grouped_feature_subsets:
            ax.set_xlabel('feature subset', fontsize = 28)
        else:
            ax.set_xlabel('feature', fontsize = 28)
    else: #don't plot the error bars
        plt.bar(range(dataset_obj.n_data_cols), dataset_obj.avg_f_importances[dataset_obj.f_imp_idxs], color="r", align="center")

    maxtick = np.around(np.max(series), decimals=1)
    print 'maxtick', maxtick
    ax.set_yticklabels(np.linspace(0, maxtick, int(maxtick*20)+1), fontsize = 20)



    plt.xticks(rotation=90)
    ax.set_xticks(range(dataset_obj.n_data_cols))
    if dataset_obj.n_f_subsets > 100:
        ax.axes.get_xaxis().set_visible(False)
    # else:
    else:
        # print 'dataset_obj.f_imp_idxs', dataset_obj.f_imp_idxs
        # print 'dataset_obj.f_subset_short_names', dataset_obj.f_subset_short_names
        # print 'dataset_obj.f_subset_short_names[dataset_obj.f_imp_idxs]', dataset_obj.f_subset_short_names[dataset_obj.f_imp_idxs]
        if not grouped_feature_subsets:
            ax.set_xticklabels([dataset_obj.f_subset_short_names[x] for x in dataset_obj.f_imp_idxs], fontsize = 20)
        else:
            ax.set_xticklabels([dataset_obj.f_subset_short_names[x] for x in dataset_obj.f_ss_imp_idxs], fontsize = 20)
        plt.tick_params(axis='both', which='major', labelsize=20)

    plt.xlim([-1, dataset_obj.n_f_subsets])
    plt.show(block = False)


    filestring = '/' + dataset_obj.img_name_string + '_FEATURE_IMPORTANCES'
    filestring = filestring.replace(" ", "_")

    if save_data:
        save_fig(experiments_folder+filestring, 'svg')
        save_fig(experiments_folder+filestring, 'png')
    plt.close()


def get_greedy_acqn_seq():
    n_unique_costs = np.unique(dataset_obj.f_subset_costs).shape[0]
    f_greedy_idxs = [[] for i in range(n_unique_costs)]

    print 'n_unique_costs', n_unique_costs

    for i, c in enumerate(np.unique(dataset_obj.f_subset_costs)):
        c_idxs = np.where(dataset_obj.f_subset_costs == c)[0]
        print 'c_idxs', c_idxs
        print 'dataset_obj.avg_fss_importances', dataset_obj.avg_fss_importances
        print 'dataset_obj.avg_fss_importances[c_idxs]', dataset_obj.avg_fss_importances[c_idxs]
        print 'np.argsort(dataset_obj.avg_fss_importances[c_idxs])[::-1]', np.argsort(dataset_obj.avg_fss_importances[c_idxs])[::-1]
        f_greedy_idxs[i] = c_idxs[np.argsort(dataset_obj.avg_fss_importances[c_idxs])[::-1]]
        print 'c_idxs', c_idxs, 'f_greedy_idxs[i] ', f_greedy_idxs[i]

    dataset_obj.f_greedy_idxs = np.hstack((f_greedy_idxs))


def load_feature_subset_names(filestring):
    with open(filestring) as f:
        featSubsetNames = []
        for subsetName in f:
            featSubsetNames.append(subsetName.rstrip())
    return featSubsetNames


def gen_generic_feat_costs(x_dimensionality):
    depths = np.ones(x_dimensionality) #feature depths
    costs = np.ones(x_dimensionality)  #load the computation times for each feature subset
    nSubsets = x_dimensionality

    return depths, costs, nSubsets


def gen_generic_feat_names(x_dimensionality):
    names = map(str, np.arange(1, x_dimensionality+1))
    shortNames = names
    return names, shortNames


def get_subplot_row_cols(scenes_per_fig):

    spRows = math.floor(math.sqrt(float(scenes_per_fig)))
    spCols = math.ceil(float(scenes_per_fig)/spRows)
    return spRows, spCols


def boxplotter(data, x_series, y_label, full_range, x_label, title_string, title_vals, x_ticks_in = None,  y_lim = None, baseline_on = False, legend_locn=1, save_prefix=''):
    #this function can be used to plot comp. times, AUCs and other stuff as a fn of sensitivity

    # images_per_fig, spRows, spCols = get_subplot_row_cols(nTestBatches)
    sp_per_fig = 6
    spRows, spCols = get_subplot_row_cols(sp_per_fig)

    total_sps = data.shape[2]
    nFigs = np.ceil(float(total_sps)/sp_per_fig).astype(int)

    # nFigs = np.ceil(float(nTestBatches)/sp_per_fig).astype(int)
    print 'boxplotter nFigs:', nFigs


    box_widths = 0.5 * np.min(x_series[1:] - x_series[:-1])     #set the widths of the boxplots to be half of the smallest distance between boxes
    print 'box_widths', box_widths
    sens_range = np.max(x_series) - np.min(x_series)

    if full_range:
        ver = '_long_'
    else:
        ver = '_short_'

    for fig_idx in range(nFigs):

        fig = figure()
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        for idx in range(sp_per_fig):

            sp_id = idx+sp_per_fig*fig_idx
            if sp_id == total_sps:
                break #break when all subplots have been plotted

            y = data[:,:,sp_id]
            ax = fig.add_subplot(spRows, spCols, idx+1)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            ax.set_title(title_string + str(title_vals[sp_id])+ ')')
            # if datasetNo in np.arange(0,2):
            #     ax.set_title('Scene: ' + str(dataset_obj.testSetScenes[sp_id])+ ', ' +  param_string)
            # else:
            #     ax.set_title(dataset_obj.fig_title_string + ' ' + param_string)

            plt.boxplot(y, positions = x_series, widths = box_widths)
            # ax.set_xticks((dataset_obj.xlabels))
            # ax.set_xticklabels((dataset_obj.xlabels))
            xlim = [np.min(x_series)-0.05*sens_range, np.max(x_series)+0.05*sens_range]
            ax.set_xlim(xlim[0],xlim[1])

            if not full_range:
                ax.set_xlim((np.min(x_series)-0.05*sens_range, dataset_obj.reduced_sens_plot_range + box_widths))

            plt.xticks(rotation=dataset_obj.xlabel_rotation)
            if x_ticks_in is not None:
                print 'x_ticks_in', x_ticks_in
                ax.set_xticks(x_ticks_in)
                ax.set_xticklabels(x_ticks_in)


            if y_lim is not None:
                ax.set_ylim((y_lim[0], y_lim[1]))

            if baseline_on:
                baseline = np.median(y[:,-1])
                print 'baseline ', baseline
                plt.plot(xlim, [baseline , baseline], 'r--', label = 'Full Dataset Baseline \n(median of '+str(final_RF_runs)+' runs)')
                plt.legend(loc = legend_locn)

        show(block = False)

        filestring = '/' + save_prefix + dataset_obj.img_name_string + str(rf_params_obj.num_trees) + 'T_sensitivity_vs_'+ y_label + ver + str(fig_idx+1) + 'of' + str(nFigs)
        filestring = filestring.replace(" ", "_")

        if save_data:
            save_fig(experiments_folder+filestring, 'svg')
            save_fig(experiments_folder+filestring, 'png')
        plt.close()


    return


def scatter_plotter(y_data, x_data, title_string, title_vals = None, y_axis=None, x_axis=None, y_lim = None, technicolor_on = False, legend_locn = 1):
    #this function can be used to plot comp. times, AUCs and other stuff as a fn of sensitivity

    sp_per_fig = 6

    total_sps = x_data.shape[2]
    nFigs = np.ceil(float(total_sps)/sp_per_fig).astype(int)
    spRows, spCols = get_subplot_row_cols(sp_per_fig)
    print 'scatter_plotter nFigs:', nFigs

    time_range = np.max(x_data)-np.min(x_data)

    if technicolor_on:
        d_colors = get_technicolor(y_data.shape[1])
        mark_style = ['o', 's', 'D']

    for fig_idx in range(nFigs):

        fig = figure()
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        for idx in range(sp_per_fig):

            sp_id = idx+sp_per_fig*fig_idx
            if sp_id  == total_sps: break

            x = x_data[:,:, sp_id]
            y = y_data[:,:, sp_id] #for each scene, ravel the auc and
            ax = fig.add_subplot(spRows, spCols, idx+1)

            if technicolor_on:
                for cfr_idx, sensitivity in enumerate(dataset_obj.cserf_sensitivities):
                    plt.scatter(x[:,cfr_idx],y[:,cfr_idx], s=20, c=d_colors[cfr_idx], label = 'sensitivity='+str(sensitivity), lw=0, marker = mark_style[cfr_idx%3])
                plt.legend(loc = legend_locn)
            else:
                plt.scatter(x,y, s=5)

            ax.set_xlabel('Test-Time Feature Cost')
            if y_axis is not None:
                ax.set_ylabel(y_axis)
            if x_axis is not None:
                ax.set_xlabel(x_axis)
            ax.set_xlim((np.min(x_data)-0.05*time_range, np.max(x_data)+0.05*time_range))

            if y_lim is not None:
                ax.set_ylim((y_lim[0], y_lim[1]))

            if title_vals is not None:
                ax.set_title(title_string + str(title_vals[sp_id]) + ')' )
            else:
                ax.set_title(title_string)


            # if datasetNo in np.arange(0,2):
            #     ax.set_title('Scene: ' + str(dataset_obj.testSetScenes[idx+sp_per_fig*fig_idx]) + ', ' +  param_string)
            # else:
            #     ax.set_title(dataset_obj.fig_title_string + ' ' + param_string)

        show(block = False)

        if save_data:
            filestring = '/' + dataset_obj.img_name_string + str(rf_params_obj.num_trees) + 'T_Test_Time_vs_' + y_axis + '_' + str(fig_idx+1) + 'of' + str(nFigs)
            filestring = filestring.replace(' ', '_')
            save_fig(experiments_folder+filestring, 'svg')
            save_fig(experiments_folder+filestring, 'png')

        plt.close()


def scatter_plotter_from_df(dataframe, time_key, y_axis_key_in, y_axis_name, pareto_on = False, pareto_data = None, baseline = None, legend_locn = 1, save_prefix = '', validation = False):
    #this function can be used to plot comp. times, AUCs and other stuff as a fn of sensitivity

    sp_per_fig = 6
    spRows, spCols = get_subplot_row_cols(sp_per_fig)

    if validation is False:
        nFigs = np.ceil(float(nTestBatches)/sp_per_fig).astype(int)
    else:
        nFigs = 1

    print 'scatter_plotter_from_df nFigs:', nFigs

    time_range = np.max(dataframe[time_key])-np.min(dataframe[time_key])

    for fig_idx in range(nFigs):

        fig = figure()
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        for idx in range(sp_per_fig):

            batch_idx = idx+sp_per_fig*fig_idx
            if  batch_idx == nTestBatches: break
            if datasetNo in range(0,2) and validation is False:
                y_axis_key = y_axis_key_in + ' B' + str(dataset_obj.testSetScenes[batch_idx])
            elif nTestBatches == 1 or validation is True:
                y_axis_key = y_axis_key_in
            else:
                y_axis_key = y_axis_key_in + ' B' + str(batch_idx)

            x = dataframe[time_key]
            y = dataframe[y_axis_key] #for each scene, ravel the auc and
            ax = fig.add_subplot(spRows, spCols, idx+1)
            plt.scatter(x,y, s=5)
            ax.set_xlabel('Test Time Feature Cost')
            ax.set_ylabel(y_axis_name)
            if datasetNo in np.arange(0,2):
                ax.set_title('Scene: ' + str(dataset_obj.testSetScenes[batch_idx]) + ', ' +  param_string)
            else:
                ax.set_title(dataset_obj.fig_title_string + ' ' + param_string)
            xlim = [np.min(x)-0.05*time_range, np.max(x)+0.05*time_range]
            ax.set_xlim(xlim [0], xlim[1])

            if (datasetNo in np.arange(2,4)) & (y_axis_key_in in ['SKL TE', 'OMA TE']):
                ax.set_ylim((dataset_obj.plot_TE_ylim[0], dataset_obj.plot_TE_ylim[1]))

            if baseline is not None:
                print 'baseline ', baseline
                plt.plot(xlim, [baseline , baseline], 'r--', label = 'Full Dataset Baseline \n(median of '+str(final_RF_runs)+' runs)')
                plt.legend(loc=legend_locn)

            if pareto_on:
                print 'pareto_data', pareto_data
                plt.plot(pareto_data[:,0], pareto_data[:,1])


        show(block = False)

        if save_data:
            filestring = '/' + save_prefix + dataset_obj.img_name_string + str(rf_params_obj.num_trees) + 'T_Test_Time_vs_' + y_axis_name + '_' + str(fig_idx+1) + 'of' + str(nFigs) + '_RUN' +str(r)
            if pareto_on:
                filestring += '_pareto'
            filestring = filestring.replace(' ', '_')
            save_fig(experiments_folder+filestring, 'svg')
            save_fig(experiments_folder+filestring, 'png')

        plt.close()


def boxplotter_multi_series(y_data, x_data, y_axis):
    #this function can be used to plot comp. times, AUCs and other stuff as a fn of sensitivity

    spRows, spCols = get_subplot_row_cols(nTestBatches)

    #create matrix of sensitivities for each matrix
    fig = figure()
    if datasetNo in range(2):
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')


    d_colors = get_technicolor(y_data.shape[0])


    for idx in range(nTestBatches):
        ax = fig.add_subplot(spRows, spCols, idx+1)
        ax.set_xlabel('Test Time Feature Cost')
        ax.set_ylabel(y_axis)

        x_diffs = x_data[:,idx,1:] - x_data[:,idx,:-1]
        print 'x_diffs', x_diffs
        box_widths = 0.5 * np.min(x_diffs[:])
        print 'box_widths ', box_widths

        for run in range(y_data.shape[0]):
            y = y_data[run,idx,:,:]
            x = x_data[run,idx,:]
            x = np.transpose(np.tile(x,(y.shape[1], 1)))
            # print 'x',x; print 'x.shape', x.shape; print 'y.shape', y.shape

            ### CAN CHOOSE EITHER BOX PLOT OR SCATTER
            # plt.scatter(x,y, color = d_colors[run])
            bp = plt.boxplot(np.transpose(y), positions = x[:,0], widths=box_widths)
            plt.setp(bp['boxes'], color=d_colors[run])
            plt.setp(bp['whiskers'], color=d_colors[run])

        if datasetNo in np.arange(0,2):
            ax.set_title('Scene: ' + str(dataset_obj.testSetScenes[idx]) + ', ' +  param_string)
        else:
            ax.set_title(dataset_obj.fig_title_string + ' ' + param_string)

        plt.close()

    show(block=False)

    if save_data:
        filestring = '/' + dataset_obj.img_name_string + str(rf_params_obj.num_trees) + 'T_Test_Time_vs_' + y_axis
        filestring = filestring.replace(' ', '_')
        save_fig(experiments_folder+filestring, 'svg')
        save_fig(experiments_folder+filestring, 'png')
    plt.close()


    show(block=False)


def calc_roc_auc(y, y_pred, prnt_str):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
    auroc = auc(false_positive_rate, true_positive_rate)
    print (prnt_str + ' AUROC: ' + str(auroc))
    return auroc


def save_fig(path, ext='svg', close=False, verbose=True):
    # Extract the directory folder and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)

    directory_path = os.getcwd() + directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    savepath = directory_path + '/' + filename     # The final path to save to
    savepath = savepath.replace(" ", "_")

    if verbose:     print("Saving figure to '%s'..." % savepath),
    plt.savefig(savepath)    # Actually save the figure
    if verbose:     print("Done")


def load_vanilla_dataset(dataset_path):

    # LOAD DATASET STORED IN SINGLE CSV FILE WITH Y VALUES IN THE FINAL COLUMN

    full_dataset = np.array(read_csv(dataset_path, header=None))
    x_full = full_dataset[:, 0:-1]
    y_full = full_dataset[:, -1]

    if 0 not in np.unique(y_full):
        max_label = np.max(np.unique(y_full))
        y_full[y_full == max_label] = 0

    val_split_idx = np.round(full_dataset.shape[0]*0.6).astype(int)
    test_split_idx = np.round(full_dataset.shape[0]*0.8).astype(int)

    x_train = x_full[:val_split_idx, :]
    y_train = y_full[:val_split_idx]
    x_val = x_full[val_split_idx:test_split_idx, :]
    y_val = y_full[val_split_idx:test_split_idx]
    x_test = [x_full[test_split_idx:, :]]
    y_test = [y_full[test_split_idx:]]

    return x_train, y_train, x_val, y_val, x_test, y_test

      #   ### the testing function expects multiple batches of test data, stored in a list of np arrays
      #   self.x_test = [np.array(read_csv(dataset_path + '/xte.csv', header=None)).transpose()]
      #   self.y_test = [np.genfromtxt(dataset_path + '/yte.csv', delimiter=',', dtype=int)]
      #   self.y_test[0][self.y_test[0]==-1]=0
      #
      #   print(' Done')
      #   # print 'self.x_train_full', self.x_train_full, 'self.x_val', self.x_val, 'self.x_test', self.x_test
      #   # print 'self.y_train_full', self.y_train_full, 'self.y_val', self.y_val, 'self.y_test', self.y_test
      #
      #   self.cserf_sensitivities = np.linspace(0, 1, 6)
      #   self.f_subset_depths, self.f_subset_costs, self.n_f_subsets = gen_generic_feat_costs(self.x_train_full.shape[1])  #load the computation times for each feature subset
      #   self.f_subset_names, self.f_subset_short_names = gen_generic_feat_names(self.x_train_full.shape[1])
      #   self.n_data_cols = np.sum(self.f_subset_depths)
      #
      #   self.cost_strip_lbs = np.linspace(0.5, 398.5, 200)
      #   self.cost_strip_ubs = np.hstack((self.cost_strip_lbs[1:], 400.5))
      #
      #   self.cserf_sensitivities = np.linspace(0,2,11)
      #   self.xlabels = np.array([0., 0.4, 0.8, 1.2, 1.6, 2.0])
      #   # self.cserf_sensitivities = np.array([0., 0.2, 0.4, 0.6, 0.64, 0.66, 0.68, 0.7, 0.72])
      #   # self.xlabels = np.array([0., 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8])
      #   self.xlabel_rotation = 90
      #
      #   # self.reduced_sens_plot_range = 0.65
      #   # self.reduced_sens_plot_AUC_ylim = [0.75, 1.0]
      #   # self.reduced_sens_plot_TE_ylim = [0.0, 0.15]
      #   self.fig_title_string = 'CIFAR10 dataset '
      #   self.img_name_string = "CIFAR10_"
      #
      #   self.plot_TE_ylim = [0.3, 0.44] # to match the Feng Nan et al. paper
      #


def gen_generic_cost_strips(n_features):

    strip_width = 2
    n_cost_strips = int(n_features/strip_width)
    cost_strip_lbs = np.linspace(0.5, n_features-(3*strip_width/4), n_cost_strips)
    cost_strip_ubs = np.hstack((cost_strip_lbs[1:], n_features+strip_width/4))

    return n_cost_strips, cost_strip_lbs, cost_strip_ubs


def generic_make_train_set(dataset_obj):

    if (2 * rf_maxSamplesPerLabel) < dataset_obj.x_train_full.shape[0]:
        rand_inds = np.random.choice(dataset_obj.x_train_full.shape[0], 2 * rf_maxSamplesPerLabel, replace=False)
        x_train = dataset_obj.x_train_full[rand_inds, :]
        y_train = dataset_obj.y_train_full[rand_inds]
    else:
        x_train = dataset_obj.x_train_full
        y_train = dataset_obj.y_train_full

    print 'xTrain', x_train , '\n', 'yTrain', y_train , '\n', 'x_test', dataset_obj.x_test , '\n', 'y_test', dataset_obj.y_test , '\n'
    print 'xTrain.shape', x_train.shape , '\n', 'yTrain.shape', y_train.shape , '\n', 'dataset_obj.x_test.shape', dataset_obj.x_test[0].shape , '\n', 'dataset_obj.y_test.shape', dataset_obj.y_test[0].shape

    return x_train, y_train


def get_technicolor(n_colours):
    HSV_tuples = [(x*1.0/n_colours, 0.5, 0.5) for x in range(n_colours)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    d_colors = []
    for tup in RGB_tuples:
        tup = tuple(i * 255 for i in tup)
        d_colors.append ('#%02x%02x%02x' % tup)
    return d_colors


def get_experiments_folder_string():
        experiments_folder = "/Results/" + dataset_obj.folder_save_prefix + algoString + time.strftime("%y%b%d-%H-%M-%S", \
                            time.localtime()) + '-Scns-'+ str(nTestBatches) + '-R-' + str(runs) + '-Tr-' + str(rf_max_num_trees) + \
                             '-SPL-' + str(rf_maxSamplesPerLabel)
        print 'experiments_folder', experiments_folder
        return experiments_folder


if __name__ == '__main__':

    ### INITIATION
    random.seed(280284)
    workingDir = 'C:\Users\l\Desktop\OCCLUSIONS\Python_RF'
    os.chdir(workingDir)

    ### ALGORITHMS: 0 = CSERF, 1 = BASIC CSR-Shrubberies, 2 = Pareto Random Shrubberies
    algoNo = 0
    ### DATASETS: 0=Lean Occlusions, 1=Full Occlusions, 2=MiniBooNE, 3=ForestCover, 4=CIFAR-10, 5=Climate Crahses, 6=Fertility, 7=Vehicle
    datasetNo = 0

    # RUN AND SAVING PARAMETERS
    runs = 25 # set to greater than one for cserf algo
    max_val_RFs_per_strip = 2
    shrub_repeats = 2
    use_fast_prototype_params = False  #In order to quickly test changes to code, can turn on a set of params for quick running of the file to
    show_features = False

    recombine_train_val_sets = False
    valset_subsample_idx = 1000000

    #NOTE: CURRENTLY ONLY BEING APPLIED TO THE FOREST COVER DATASET
    grouped_feature_subsets = True

    #SAVING OPTIONS
    save_data = True
    save_stacked_bar_graphs = True

    #ALTERATIVE SCHEMES
    compare_with_rand_subsets = True
    compare_with_greedy_algo = True

    #PRINTING OPTIONS
    verbose_cserfs = True


    # RF CLASSIFIER SELECTION
    useSKLForest = True
    useOMAForest = False
    printTestTimes = False

    if algoNo == 0 and not useSKLForest and not useOMAForest:
        print 'Must set either or both of useSKLForest and useOMAForest to True if running linear penalty CSRF algorithm!'
        sys.exit()

    # RF GENERAL SETTINGS (shared by OMA, SKL, and Basic CS RFs). !!!NOTE!!! There is an "override settings" option within some datasets that change these!
    rf_maxSamplesPerLabel = 2000 # 7000 is the amount used by Ahmad in his code. He found that any number greater than 400 gave close to the best performance
    rf_max_num_trees = 100
    rf_max_depth = 6
    rf_min_sample_count = 20
    rshrub_train_pllel = False
    rf_criterion = 'gini'
    final_RF_runs = 5

    ### FAST PROTOTYPING PARAMETERS GO HERE
    if use_fast_prototype_params:

        if algoNo == 0:runs = 2
        if algoNo == 1:runs = 1 #setting to one run will crash the thing

        shrub_repeats = 2
        rf_maxSamplesPerLabel = 250 # 7000 is the amount used by Ahmad in his code. He found that any number greater than 400 gave close to the best performance
        rf_max_num_trees = 100
        valset_subsample_idx = 6000
        final_train_abort_loop = 5

    ### TREE NUMBERS AT WHICH TO SAMPLE FROM LINEAR PENALTY COST SENSITIVE RFs
    cserf_sample_points = np.array([1,4,8,20,40,100])  # powers of 2 seems like a good idea (remember python indexing starts at 0!)
    unary_cost_dset_sensitivities = np.array([0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.20])
    # unary_cost_dset_sensitivities = np.array([0, 0.015, 0.03, 0.045, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24])
    unary_cost_dset_sens_xticks = np.linspace(0, np.max(unary_cost_dset_sensitivities ), 9)

    # cserf_sample_points = np.array([1,2,4,8,16,32,48,64,80])  # powers of 2 seems like a good idea (remember python indexing starts at 0!)
    cserf_sample_points = cserf_sample_points[cserf_sample_points <= rf_max_num_trees] #remove those higher than our max num of trees
    n_cserf_samples = len(cserf_sample_points)
    print 'cserf_sample_points ', cserf_sample_points, ', n_cserf_samples', n_cserf_samples

    ### DATASET SETUP - CALLS FUNCTION THAT CREATES APPROPRIATE DATASET OBJECT
    dataset_obj, nTestBatches, rf_num_classes = make_dataset_object(datasetNo, use_fast_prototype_params, grouped_feature_subsets)

    ### RF GENERAL PARAMETERS SETUP. Creates struct which is passed into the various RFs used in this project.
    rf_params_obj = rf.ForestParams(num_classes=rf_num_classes, trees=rf_max_num_trees, depth=rf_max_depth)

    ### STRING FOR ADDING TO GRAPH TITLES
    param_string = 'nTree=' + str(rf_max_num_trees)
    if algoNo == 0:         algoString = '_cserf_'
    if algoNo == 1:         algoString = '_shrub-srch-RF_reps-' + str(shrub_repeats) + '_'

    ### MAKE TRAINING, VALIDATION AND TEST SET
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_obj.make_train_test_set() #produces train set of with rf_maxSamplesPerLabel pixels per class


    ### TRAIN A BENCHMARK SKL ON THE FULL TRAINING SET
    rf_no_active_vars = np.sqrt(x_train.shape[1]).astype(int) # used only by the SKL forest
    print 'rf_no_active_vars', rf_no_active_vars
    benchmark_RFs = [RandomForestClassifier(n_estimators=rf_max_num_trees, criterion=rf_criterion, max_depth=rf_max_depth, min_samples_split=rf_min_sample_count,
                                            max_features=rf_no_active_vars, oob_score=False) for r in range(final_RF_runs)]
    for r, forest in enumerate(benchmark_RFs):
        tic = time.time()
        forest.fit(x_train, y_train)
        print 'Training SKL benchmark classifier', r,', Train time:', time.time()-tic


    ### DETERMINE THE NAME OF NEW FOLDER TO SAVE EXPERIMENTS INTO
    experiments_folder = get_experiments_folder_string()
    ### GET THE FEATURE IMPORTANCES
    get_avg_f_importances(benchmark_RFs, experiments_folder)




    ### GET GREEDY ACQUISITION SEQUENCE IF REQUIRED
    if compare_with_greedy_algo & (np.unique(dataset_obj.f_subset_costs).shape[0] > 1):
        get_greedy_acqn_seq()


    ### INITIALISE ALGORITHM
    # if algoNo == 0:
    cserf_mgr = CSERF_Manager(dataset_obj.cserf_sensitivities, benchmark_RFs)

    ### (optional): show features alongside the GT occlusion in separate window
    if show_features:
        scene = min((dataset_obj.displayScene, dataset_obj.maxNoTrainScenes))-1
        # scene = min((cserf_mgr.displayScene, cserf_mgr.maxNoTrainScenes))-1
        featVectors = dataset_obj.allTrainFVs[scene]
        groundTruth = dataset_obj.allTrainGTs[scene]
        print 'displaying features for scene number ', scene
        print 'featVectors.shape', featVectors.shape
        print 'groundTruth.shape', groundTruth.shape
        dataset_obj.show_features(x = featVectors, y = groundTruth, experiments_folder=experiments_folder, sceneNo=dataset_obj.testSetScenes[scene]) # will plot all of the features using matplotlib.pyplot


    ### CLEAN UP MEMORY
    del dataset_obj.allTrainFVs, dataset_obj.allTrainGTs # delete the done-with feature vector and ground truth matrices which take up GBs of memory


    # if algoNo == 1:
    #     basic_shrub_srch = BasicShrubberySearchManager(n_repeats = shrub_repeats, n_forest_positions=3)


    ### CSERFs / SHRUBBERIES TRAINING LOOP
    for r in range(0,runs):

        if r == 0: param_string = param_string + 'Train Set: ' + str(x_train.shape[0]) + ' samples' # graph titles will display the training set size in

        ### CREATE AND TRAIN FAMILY OF Linear Penanlty CSRFs
        if algoNo == 0:
            cserf_mgr.grow_CSERFs(dataset_obj, x_train, y_train, x_val, y_val, runs, recombine_train_val_sets)
            cserf_mgr.predict_with_CSERFs(x_val, y_val)
            cserf_mgr.get_val_RFs_candidates_from_cserfs()

        elif algoNo == 1:
            basic_shrub_srch.plant_shrubberies(r, x_train, y_train, x_val, y_val)
            basic_shrub_srch.grow_CSERFs(dataset_obj, x_train, y_train, x_val, y_val, r, recombine_train_val_sets)
            # print 'shrubbery_subsets', shrubbery_subsets; print 'x_tr_shrub_cols', x_tr_shrub_cols; print 'used_f_sh_names', used_f_sh_names

            ### TEST THE IMAGES
            for batch_idx in range(len(y_test)):

                basic_shrub_srch.predict_with_forests(x_test[batch_idx], y_test[batch_idx], batch_idx)





    ## don't delete this!!
    if algoNo == 0:
        cserf_mgr.analyse_CSERFs()

    ## PREDICT WITH Lin Pen CSRF RF CLASSIFIERS
    if algoNo == 0:

        cserf_mgr.collate_val_RFs_cands()
        cserf_mgr.train_test_val_RFs(x_train, y_train, x_val, y_val, experiments_folder)
        cserf_mgr.identify_final_forests()
        cserf_mgr.train_final_forests(x_train, y_train, final_RF_runs)

        ### TEST THE FINAL FORESTS ON THE
        for batch_idx in range(len(y_test)):
            cserf_mgr.test_final_forests(x_test[batch_idx], y_test[batch_idx], batch_idx)




    if algoNo == 1:
        basic_shrub_srch.analyse_forests()


    if algoNo == 0:

        cserf_mgr.build_final_results_df(experiments_folder)
        cserf_mgr.analyse_final_forests()

        if compare_with_rand_subsets:
            cserf_mgr.create_random_val_RFs()
            cserf_mgr.train_test_val_RFs(x_train, y_train, x_val, y_val, experiments_folder, save_prefix = "RANDOM_")
            cserf_mgr.identify_final_forests(save_prefix = 'RANDOM_')
            cserf_mgr.train_final_forests(x_train, y_train, final_RF_runs)

            ### TEST THE FINAL FORESTS ON THE
            for batch_idx in range(len(y_test)):
                cserf_mgr.test_final_forests(x_test[batch_idx], y_test[batch_idx], batch_idx, save_prefix = 'RANDOM_')

            cserf_mgr.build_final_results_df(experiments_folder, save_prefix = 'RANDOM_')
            cserf_mgr.analyse_final_forests(save_prefix = 'RANDOM_')

        if compare_with_greedy_algo:
            cserf_mgr.greedy_val_RFs()
            cserf_mgr.train_test_val_RFs(x_train, y_train, x_val, y_val, experiments_folder, save_prefix = "GREEDY_")
            cserf_mgr.identify_final_forests(save_prefix = 'GREEDY_')
            cserf_mgr.train_final_forests(x_train, y_train, final_RF_runs, )

            ### TEST THE FINAL FORESTS ON THE
            for batch_idx in range(len(y_test)):
                cserf_mgr.test_final_forests(x_test[batch_idx], y_test[batch_idx], batch_idx, save_prefix = 'GREEDY_')

            cserf_mgr.build_final_results_df(experiments_folder, save_prefix = 'GREEDY_')
            cserf_mgr.analyse_final_forests(save_prefix = 'GREEDY_')

    print("Lin Pen CSRF RF Completed Successfully")






