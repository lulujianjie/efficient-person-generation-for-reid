import pandas as pd
import numpy as np
class Market1501Pose():
    def __init__(self, data_dir = 'data_dir', verbose = True, split = 'train', restore = 'True'):
        super(Market1501Pose, self).__init__()
        self.dataset_dir = data_dir
        self.split = split
        self.restore = restore
        if self.restore:
            self.datalist = np.load(self.dataset_dir+'/part/{}.npy'.format(self.split))
            if verbose:
                print("=> Loaded from npy")
                print("=> Market1501-Pose loaded with {} {} pairs".format(len(self.datalist), self.split))
        else:
            if self.split == 'train':
                train = self.get_path_list(self.dataset_dir, self.split)
                if verbose:
                    print("=> Market1501-Pose loaded with {} training pairs".format(len(train)))

                self.datalist = train
            else:
                test = self.get_path_list(self.dataset_dir, self.split)

                if verbose:
                    print("=> Market1501-Pose loaded with {} test pairs".format(len(test)))

                self.datalist = test
            np.save(self.dataset_dir+'/pose/{}.npy'.format(self.split), self.datalist)

    def get_path_list(self, data_dir, split):
        dataset = []
        if split == 'train':
            dataset_pair = pd.read_csv(data_dir+'market-pairs-train.csv')
            print('=>Processing train data...')
            for i in range(len(dataset_pair)):
                img_path1 = data_dir+'bounding_box_train/'+dataset_pair.iloc[i]['from']
                img_path2 = data_dir+'bounding_box_train/'+dataset_pair.iloc[i]['to']
                pose_heatmap_path1 = data_dir + 'train_part_heatmap/' + dataset_pair.iloc[i]['from']+'.npy'
                pose_heatmap_path2 = data_dir + 'train_part_heatmap/' + dataset_pair.iloc[i]['to'] + '.npy'
                dataset.append((img_path1, pose_heatmap_path1, img_path2, pose_heatmap_path2))
        else:
            dataset_pair = pd.read_csv(data_dir+'market-pairs-test.csv')
            print('=>processing test data...')
            for i in range(len(dataset_pair)):
                img_path1 = data_dir+'bounding_box_test/'+dataset_pair.iloc[i]['from']
                img_path2 = data_dir+'bounding_box_test/'+dataset_pair.iloc[i]['to']
                pose_heatmap_path1 = data_dir + 'test_part_heatmap/' + dataset_pair.iloc[i]['from']+'.npy'
                pose_heatmap_path2 = data_dir + 'test_part_heatmap/' + dataset_pair.iloc[i]['to'] + '.npy'
                dataset.append((img_path1, pose_heatmap_path1, img_path2, pose_heatmap_path2))
        return dataset


