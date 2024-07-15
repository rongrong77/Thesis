import numpy as np
import pandas as pd
from loading.loadpickledataset import LoadPickleDataSet
from preprocessing.filter_imu import FilterIMU
from preprocessing.filter_opensim import FilterOpenSim
from preprocessing.remove_outlier import remove_outlier
from preprocessing.resample import Resample
from preprocessing.segmentation.fixwindowsegmentation import FixWindowSegmentation


class DataSet:
    def __init__(self, config, load_dataset=True):
        self.config = config
        self.x = []
        self.y = []
        self.labels = []
        self.selected_trial_type = config['selected_trial_type']
        self.segmentation_method = config['segmentation_method']
        self.resample = config['resample']
        self.n_sample = len(self.y)
        if load_dataset:
            self.load_dataset()
            self.train_subjects = config['train_subjects']
            self.test_subjects = config['test_subjects']
            self.train_activity = config['train_activity']
            self.test_activity = config['test_activity']
        self.train_dataset = {}
        self.test_dataset = {}

    def load_dataset(self):
        getdata_handler = LoadPickleDataSet(self.config)
        self.x, self.y, self.labels = getdata_handler.run_get_dataset()
        self._preprocess()

    def _preprocess(self):
        # self.x, self.y, self.labels = remove_outlier(self.x, self.y, self.labels)
        # print("After removing outliers: ", len(self.x), len(self.y),len(self.labels))
        # # if self.resample:
        # #     self.x, self.y, self.labels = self.run_resample_signal(self.x, self.y, self.labels)
        # #     print("After resampling: ", len(self.x), len(self.y))
        if self.config['opensim_filter']:
            filteropensim_handler = FilterOpenSim(self.y, lowcut=6, fs=100, order=2)
            self.y = filteropensim_handler.run_lowpass_filter()
        if self.config['imu_filter']:
            filterimu_handler = FilterIMU(self.x, lowcut=10, fs=100, order=2)
            self.x = filterimu_handler.run_lowpass_filter()

    def run_resample_signal(self, x, y, labels):
        resample_handler = Resample(x, y, labels, 200, 100)
        x, y, labels = resample_handler._run_resample()
        return x, y, labels

    def run_segmentation(self, x, y, labels):
        if self.segmentation_method == 'fixedwindow':
            segmentation_handler = FixWindowSegmentation(x, y, labels, winsize=self.config['target_padding_length'], overlap=0.5)
            self.x, self.y, self.labels = segmentation_handler._run_segmentation()

        if self.config['opensim_filter']:
            filteropensim_handler = FilterOpenSim(self.y, lowcut=6, fs=200, order=2)
            self.y = filteropensim_handler.run_lowpass_filter()

        # print("Segmentation results:")
        # print("x shape:", [seg.shape for seg in self.x])
        # print("y shape:", [seg.shape for seg in self.y])
        # print("labels shape:", self.labels.shape)
        return self.x, self.y, self.labels

    
    def concatenate_data(self):
        self.labels = pd.concat(self.labels, axis=0, ignore_index = True)
        self.x = np.concatenate(self.x, axis=0)
        self.y = np.concatenate(self.y, axis=0)

   
    def run_dataset_split(self):
        if set(self.test_subjects).issubset(self.train_subjects):
             train_labels = self.labels[~self.labels['subject'].isin(self.test_subjects)]
             test_labels = self.labels[(self.labels['subject'].isin(self.test_subjects))]
        else:
             train_labels = self.labels[self.labels['subject'].isin(self.train_subjects)]
             test_labels = self.labels[(self.labels['subject'].isin(self.test_subjects))]
        print(train_labels['subject'].unique())
        print(test_labels['subject'].unique())

    

        train_index = train_labels.index.values
        test_index = test_labels.index.values
        print('training length', len(train_index))
        print('test length', len(test_index))

        train_x = [self.x[i] for i in train_index]
        train_y = [self.y[i] for i in train_index]
        # self.train_dataset['x'] = train_x.reshape([int(train_x.shape[0]/self.config['target_padding_length']), self.config['target_padding_length'], train_x.shape[1]])
        # self.train_dataset['y'] = train_y.reshape([int(train_y.shape[0]/self.config['target_padding_length']), self.config['target_padding_length'], train_y.shape[1]])
        self.train_dataset['x'] = train_x
        self.train_dataset['y'] = train_y
        self.train_dataset['labels'] = train_labels.reset_index(drop=True)

        test_x = [self.x[i] for i in test_index]
        test_y = [self.y[i] for i in test_index]
        # self.test_dataset['x'] = test_x.reshape([int(test_x.shape[0]/self.config['target_padding_length']), self.config['target_padding_length'], test_x.shape[1]])
        # self.test_dataset['y'] = test_y.reshape([int(test_y.shape[0]/self.config['target_padding_length']), self.config['target_padding_length'], test_y.shape[1]])
        self.test_dataset['x'] = test_x
        self.test_dataset['y'] = test_y
        self.test_dataset['labels'] = test_labels.reset_index(drop=True)

        del train_labels, test_labels, train_x, train_y, test_x, test_y
        return self.train_dataset,  self.test_dataset


