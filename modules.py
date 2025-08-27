import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import scipy.io
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from sklearn import manifold
from dataloader import *
from torch.utils.data import TensorDataset, DataLoader

def z_score(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z = (mean - X) / (std+0.000000001)

    return z, mean, std

def normalize(X, mean, std):
    z = (mean - X) / (std+0.0000001)
    return z

def one_hot(y, n_cls):
    y_new = []
    y = np.array(y, 'int32')
    for i in range(len(y)):
        target = [0] * n_cls
        target[y[i]] = 1
        y_new.append(target)
    return np.array(y_new, 'int32')

# Obtaining TRAIN and TEST from DATA
def split_data(X, Y, seed, test_size=0.3):

    s = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in s.split(X, Y):
        X_tr, X_ts = X[train_index], X[test_index]
        Y_tr, Y_ts = Y[train_index], Y[test_index]

    return X_tr, Y_tr, X_ts, Y_ts



# dataset definition
class PseudoLabeledData(Dataset):
    # load the dataset
    def __init__(self, X, Y, W):
        self.X = torch.Tensor(X).float()
        self.Y = torch.Tensor(Y).long()
        # weights
        self.W = torch.Tensor(W).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx], self.W[idx]]
    

def load_seed(args, path, session="all", feature="LDS", n_samples=185):
    import scipy.io
    import numpy as np

    session1 = ["1_20131027", "2_20140404", "3_20140603", "4_20140621", "5_20140411", 
                "6_20130712", "7_20131027", "8_20140511", "9_20140620", "10_20131130", 
                "11_20140618", "12_20131127", "13_20140527", "14_20140601", "15_20130709"]
    
    session2 = ["1_20131030", "2_20140413", "3_20140611", "4_20140702", "5_20140418", 
                "6_20131016", "7_20131030", "8_20140514", "9_20140627", "10_20131204",  
                "11_20140625", "12_20131201", "13_20140603", "14_20140615", "15_20131016"]
    
    session3 = ["1_20131107", "2_20140419", "3_20140629", "4_20140705", "5_20140506", 
                "6_20131113", "7_20131106", "8_20140521", "9_20140704", "10_20131211",
                "11_20140630", "12_20131207", "13_20140610", "14_20140627", "15_20131105"]

    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session2
    elif session == 3:
        x_session = session3
    else:
        raise ValueError("Session must be 1, 2, or 3")

    # Load labels
    y_session = scipy.io.loadmat(path + "label.mat", mat_dtype=True)["label"][0]
    y_session = y_session + 1  # To [1, 2, 3]
    
    X_subjects = {}
    Y_subjects = {}
    
    for subj_idx, subj in enumerate(x_session):
        print("Subject load:", subj)
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)

        subj_X_list = []
        subj_Y_list = []
        
        for trial_idx in range(15):
            features = dataMat[feature + str(trial_idx + 1)]  # shape: (T, 62, 5)
            features = np.swapaxes(features, 0, 1)  # shape: (T, 62, 5)

            if features.shape[0] > n_samples:
                features = features[-n_samples:]  # keep last n samples

            # Sliding window
            window_size = 12
            temp_feats = [np.expand_dims(features[i:i + window_size], axis=0) 
                          for i in range(len(features) - window_size + 1)]
            temp_feats = np.concatenate(temp_feats, axis=0)  # shape: (N, 12, 62, 5)

            labels = np.array([y_session[trial_idx]] * temp_feats.shape[0])
            subj_X_list.append(temp_feats)
            subj_Y_list.append(labels)
        
        # Once per subject
        X_subjects[subj_idx] = np.concatenate(subj_X_list, axis=0)
        Y_subjects[subj_idx] = np.concatenate(subj_Y_list, axis=0)
        print(f"Subject {subj_idx+1}: {X_subjects[subj_idx].shape}, Labels: {Y_subjects[subj_idx].shape}")
    
    trg_subj = args.target - 1
    Tx = X_subjects[trg_subj]
    Ty = Y_subjects[trg_subj]
    Tx, m, std = z_score(Tx)

    # Train loader
    train_loader = UnalignedDataLoader()
    train_loader.initialize(len(x_session), X_subjects, Y_subjects, Tx, Ty, trg_subj,
                            args.batch_size, args.batch_size,
                            shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()

    # Test loader
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()

    return datasets, dataset_test, X_subjects, Y_subjects




def load_seed_raw(args, path, session="all", feature="LDS", n_samples=185):
    """
    SEED I
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    """
    
    

         
    
    
    # Load samples
    samples_by_subject = 0
    X = []
    Y = []
    flag = False
    X_subjects = {}
    Y_subjects = {}
    n = 15*185
    r = 0
    for subj in range(15):
        save_dir_data = f"/home/lyc/research/research_6/data_cv5fold/S{subj+1}_session{session}.npy"
        save_dir_label = f"/home/lyc/research/research_6/data_cv5fold//S{subj+1}_session{session}_label.npy"
        dataMat = np.load(save_dir_data, allow_pickle=True)
        labelMat = np.load(save_dir_label, allow_pickle=True)
        print("Subject load:", subj)

        trial_data = dataMat.reshape(-1, 62, 200)
        trial_label = np.squeeze(labelMat.reshape(-1))
        trial_label[trial_label == -1] = 2

        X_subjects[subj] = trial_data
        Y_subjects[subj] = trial_label
        # increment rang
        print(X_subjects[subj].shape)
  
    trg_subj = args.target - 1
    Tx = np.array(X_subjects[trg_subj])
    Ty = np.array(Y_subjects[trg_subj])   
    subject_ids = X_subjects.keys()
    num_domains = len(subject_ids)
    Tx, m, std = z_score(Tx)    
    # Train dataset
    train_loader = UnalignedDataLoader()
    train_loader.initialize(num_domains, X_subjects, Y_subjects, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()
    #classes = np.unique(Ty)
    # Test dataset
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()
    
    return datasets, dataset_test



def load_seed_three_feature(args, path, session="all", feature="LDS", n_samples=185):
    """
    SEED I
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    """
    
    
    session1 = [
        "1_20131027",
        "2_20140404", 
        "3_20140603", 
        "4_20140621", 
        "5_20140411", 
        "6_20130712", 
        "7_20131027",
        "8_20140511",
        "9_20140620",
        "10_20131130", 
        "11_20140618",
        "12_20131127",
        "13_20140527", 
        "14_20140601", 
        "15_20130709"
        ]
        
    session2 = [
        "1_20131030", 
        "2_20140413", 
        "3_20140611", 
        "4_20140702",
        "5_20140418",  
        "6_20131016", 
        "7_20131030", 
        "8_20140514", 
        "9_20140627", 
        "10_20131204",  
        "11_20140625",
        "12_20131201", 
        "13_20140603", 
        "14_20140615",
        "15_20131016",
        ]
        
    # SESSION 3
    
    session3 = [
        "1_20131107",
        "2_20140419",
        "3_20140629",
        "4_20140705",
        "5_20140506", 
        "6_20131113",
        "7_20131106",
        "8_20140521",
        "9_20140704",
        "10_20131211",
        "11_20140630",
        "12_20131207",
        "13_20140610", 
        "14_20140627",
        "15_20131105"
        ]
        
    feature_2 = 'psd_LDS'
    feature_3 = 'PLV'    

    # LABELS
    labels = scipy.io.loadmat(path + "label.mat", mat_dtype=True)
    y_session = labels["label"][0]
    # relabel to neural networks [0,1,2]
    for i in range(len(y_session)):
        y_session[i] += 1
    print(y_session)
    
    # select session
    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session2
    elif session == 3:
        x_session = session3
    
    # Load samples
    samples_by_subject = 0
    X = []
    plv = []
    Y = []
    flag = False
    contact = False
    index = 0
    for subj in x_session:
        # load data .mat
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)
        plv_dir_data = f"/home/lyc/dataset/seed_plv//S{index+1}_session{session}_plv.npy"
        plvMat = np.load(plv_dir_data, allow_pickle=True)        
        # psdMat = scipy.io.loadmat(path + subj + "_PSD.mat", mat_dtype=True)
        print("Subject load:", subj)
        plv_feature = plvMat.reshape(-1,62,5)
        # print("plv_feature:", plv_feature.shape)
        if contact == 0:
                plv = plv_feature 
                contact = True
        else:
                plv = np.concatenate((plv, plv_feature), axis=0)
               
        # print("plv:", plv.shape)
        index += 1
        for i in range(15):

            # "Differential_entropy (DE)"
            #   62 channels
            #   42 epochs
            #   5 frequency band
            features = dataMat[feature+str(i+1)]
            PSD_feature = dataMat[feature_2+str(i+1)]
            # [1D]
            features = np.swapaxes(features, 0, 1)
            PSD_feature = np.swapaxes(PSD_feature, 0, 1)
            # [select last 'n_samples' samples]
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]
                PSD_feature = PSD_feature[pos:]


            # [Build temporal samples]
            # + ++ + + + + + + + + ++  +

            # feats = features
            # window_size = 9
            # temp_feats = None
            # b = False
            # for a in range(len(feats) - window_size + 1):
            #     f = feats[a:a+window_size]
            #     f = np.expand_dims(f, axis=0)
            #     if not b:
            #         temp_feats = f
            #         b = True
            #     else:
            #         temp_feats = np.concatenate((temp_feats, f), axis=0)
            features = np.stack([features, PSD_feature], axis=1)
            # ++ + ++ + + + + + ++ +

            # set labels for each epoch
            labels = np.array([y_session[i]] * features.shape[0])
            # print("labels:", labels.shape)
            # print("features:", features.shape)
            
            # add to arrays
            if flag == 0:
                X = features
                Y = labels
                flag = True
            else:
                X = np.concatenate((X, features), axis=0)
                Y = np.concatenate((Y, labels), axis=0)
        
        if samples_by_subject == 0:
            samples_by_subject = len(X)
    print("X:", X.shape)
    print("plv:", plv.shape)
    plv = plv[:, np.newaxis, :, :]  # shape: [100, 1, 62, 5]
    # plv = plv[:, np.newaxis, :, :]  # shape: [100, 1, 62, 5]
    X = np.concatenate([X, plv], axis=1)  # shape: [100, 3, 62, 5]
    zero_pad = np.zeros((X.shape[0], 1, 62, 5), dtype=X.dtype)
    X = np.concatenate([X, zero_pad], axis=1)
    # reorder data by subject
    X_subjects = {}
    Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r+n]
        Y_subjects[subj] = Y[r:r+n]
        # increment range
        r += n
        print(X_subjects[subj].shape)
    trg_subj = args.target - 1
    Tx = np.array(X_subjects[trg_subj])
    Ty = np.array(Y_subjects[trg_subj])   
    subject_ids = X_subjects.keys()
    num_domains = len(subject_ids)
    Tx, m, std = z_score(Tx)    
    # Train dataset
    train_loader = UnalignedDataLoader()
    train_loader.initialize(num_domains, X_subjects, Y_subjects, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()
    #classes = np.unique(Ty)
    # Test dataset
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()
    
    return datasets, dataset_test, X_subjects, Y_subjects



def fine_tuning_load_XY(args, X, Y):
    dset_loaders = {}
    if args.dataset in ["seed", "seed-iv"]:
        print("DATA:", args.dataset, " SESSION:", args.session)
        subjects = X.keys()
        print(subjects)
        Sx = Sy = None
        i = 0
        flag = False
        selected_subject = args.target - 1
        trg_subj = -1

        for s in subjects:
            if i != selected_subject:

                tr_x = np.array(X[s])
                tr_y = np.array(Y[s])
                tr_x, m, std = z_score(tr_x)
                if not flag:
                    Sx = tr_x
                    Sy = tr_y
                    flag = True
                else:
                    Sx = np.concatenate((Sx, tr_x), axis=0)
                    Sy = np.concatenate((Sy, tr_y), axis=0)
            else:
                # store ID
                trg_subj = s
            i += 1

        print("[+] Target subject:", trg_subj)
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])
        Vx = Tx
        Vy = Ty
        Tx, m, sd = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=sd)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Vx.shape, "Ty_test:", Vy.shape)
        Sx_tensor = torch.tensor(Sx)
        Sy_tensor = torch.tensor(Sy)

        # create containers for source data
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)
        Vx_tensor = torch.tensor(Vx)
        Vy_tensor = torch.tensor(Vy)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

        # data loader
        dset_loaders["source"] = DataLoader(source_tr, batch_size=args.batch_size_fine, shuffle=True, num_workers=0, drop_last=True)
        dset_loaders["target"] = DataLoader(target_ts, batch_size=args.batch_size_fine, shuffle=True, num_workers=0, drop_last=True)
        dset_loaders["test"] = DataLoader(target_ts, batch_size=200, shuffle=False, num_workers=0)

        print("Data were succesfully loaded")

    else:
        print("This dataset does not exist.")
        exit()
    return dset_loaders