import os
import pickle
import pandas as pd


class LoadPickleDataSet:
    def __init__(self, config):
        '''
        This class loads and processes a dataset from a pickle file. The class takes a configuration dictionary as input,
        which contains various parameters such as the file path, the name of the dataset, and the selected sensors, selected_imu_features, and selected_opensim_labels.
        :param config:
        '''
        self.dl_dataset_path = config['dl_dataset_path']
        self.dataset_name = config['dl_dataset']
        self.selected_sensors = config['selected_sensors']
        self.selected_imu_features = config['selected_imu_features']
        self.selected_opensim_labels = config['selected_opensim_labels']
        self.selected_labels =  config['selected_trial_type']
        self.dataset = {}
        self.imu = None
        self.ia = None

    def load_dataset(self):
        dataset_file = os.path.join(self.dl_dataset_path + self.dataset_name)
        if os.path.isfile(dataset_file):
            print('file exist')
            with open(dataset_file, 'rb') as f:
                self.dataset = pickle.load(f)
                print(f"Loaded dataset keys: {self.dataset.keys()}")
                if 'imu' in self.dataset:
                    print(f"IMU data type: {type(self.dataset['imu'])}")
                else:
                    print("IMU data is missing from the dataset.")
                if 'ia' in self.dataset:
                    print(f"IA data type: {type(self.dataset['ia'])}")
                else:
                    print("IA data is missing from the dataset.")
        else:
            print('this dataset is not exist: run run_dataset_prepration.py first')

    def combine_sensors_features(self):
        self.selected_sensor_features = []
        for sensor in self.selected_sensors:
            ss = [sensor + '_' + imu_feature for imu_feature in self.selected_imu_features]
            self.selected_sensor_features = self.selected_sensor_features + ss

    def get_selected_ia(self):
        ia = self.dataset.get('ia', None)
        self.ia = []
        for subject, activities in ia.items():
            for activity, df_list in activities.items():
                if isinstance(df_list, list):
                    for df in df_list:
                        if isinstance(df, pd.DataFrame):
                            missing_labels = [label for label in self.selected_opensim_labels if label not in df.columns]
                            if not missing_labels:
                                self.ia.append(df[self.selected_opensim_labels].values)
                            else:
                                print(f"Missing labels in activity {activity} for subject {subject}: {missing_labels}")
                        else:
                            print(f"Expected DataFrame, got {type(df)} for activity {activity}")
                else:
                    print(f"Expected list, got {type(df_list)} for activity {activity}")

        return self.ia

    def get_selected_imu(self):
        imu = self.dataset.get('imu', None)
        self.combine_sensors_features()
        print("Selected Sensor Features:", self.selected_sensor_features)
        self.imu = []
        labels_data = []
        for subject, activities in imu.items():
            for activity, df_list in activities.items():
                if isinstance(df_list, list):
                    for df in df_list:
                        if isinstance(df, pd.DataFrame):
                            missing_labels = [label for label in self.selected_sensor_features if label not in df.columns]
                            if not missing_labels:
                                self.imu.append(df[self.selected_sensor_features].values)
                                labels_data.append({'subject': subject, 'activity': activity})
                            else:
                                print(f"Missing labels in activity {activity} for subject {subject}: {missing_labels}")
                        else:
                            print(f"Expected DataFrame, got {type(df)} for activity {activity}")
                else:
                    print(f"Expected list, got {type(df_list)} for activity {activity}")
        self.labels = pd.DataFrame(labels_data)
        return self.imu, self.labels
    
    def run_get_dataset(self):
        self.load_dataset()
        self.get_selected_imu()
        self.get_selected_ia()
        selected_y_values = self.ia
        selected_x_values = self.imu
        selected_labels =  self.labels
        del self.dataset
        return selected_x_values, selected_y_values, selected_labels


