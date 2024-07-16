import wandb
import torch
import numpy as np
import os
from torch.utils.data.dataloader import DataLoader
from config import get_config_universal
from dataset import DataSet
from datasetbuilder import DataSetBuilder
from evaluation import Evaluation
from parametertuning import ParameterTuning
from test import Test
from train import Train
from utils.utils import get_activity_index_test, get_model_name_from_activites
from visualization.wandb_plot import wandb_plotly_true_pred
import pickle
from model.transformer_tsai import TransformerTSAI
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



wandb.init(project='OEMAL_Dataset_Transformer')


def run_main():
    torch.manual_seed(0)
    np.random.seed(42)
    dataset_name = 'oemal'
    config = get_config_universal(dataset_name)
    load_model = config['load_model']
    save_model = config['save_model']
    tuning = config['tuning']
    individual_plot = config['individual_plot']
    # build and split dataset to training and test
    dataset_handler = DataSet(config, load_dataset=True)

    kihadataset_train, kihadataset_test = dataset_handler.run_dataset_split()

    train_size = 0.8
    num_train = int(train_size * len(kihadataset_train['x']))
    train_indices = np.random.choice(range(len(kihadataset_train['x'])), size=num_train, replace=False)
    val_indices = list(set(range(len(kihadataset_train['x']))) - set(train_indices))
    
    kihadataset_val = {
        'x': [kihadataset_train['x'][i] for i in val_indices],
        'y': [kihadataset_train['y'][i] for i in val_indices],
        'labels': kihadataset_train['labels'].iloc[val_indices].reset_index(drop=True)
    }
    kihadataset_train = {
        'x': [kihadataset_train['x'][i] for i in train_indices],
        'y': [kihadataset_train['y'][i] for i in train_indices],
        'labels': kihadataset_train['labels'].iloc[train_indices].reset_index(drop=True)
    }

    
    
    kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'] = dataset_handler.run_segmentation(kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'])
    kihadataset_val['x'], kihadataset_val['y'], kihadataset_val['labels'] = dataset_handler.run_segmentation(kihadataset_val['x'], kihadataset_val['y'], kihadataset_val['labels'])
    kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'] = dataset_handler.run_segmentation(kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'])
   
    if tuning == True:
        del kihadataset_test
        ParameterTuning(config, kihadataset_train)
        
    else:
        if config['model_name'] == 'transformer':
                
            # Transformer model
            c_in = kihadataset_train['x'].shape[2]
            c_out = kihadataset_train['y'].shape[2]
            seq_len = kihadataset_train['x'].shape[1]
            transformer_model = TransformerTSAI(c_in=c_in, c_out=c_out, seq_len=seq_len, classification=False)
            transformer_model.float()
            # Training the transformer model
            model_file = 'transformer_model.pt'
            
            if load_model and os.path.isfile('./caches/trained_model/' + model_file):
                print("Loading transformer model")
                transformer_model.load_state_dict(torch.load(os.path.join('./caches/trained_model/', model_file)))
            else:
                # Training the transformer model
                optimizer = torch.optim.Adam(transformer_model.parameters(), lr=config['learning_rate'])
                criterion = torch.nn.MSELoss()
                num_epochs = config['n_epoch']
                train_losses = []
                val_losses = []

                # Prepare the dataloaders
                train_dataset = DataSetBuilder(kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'],
                                            transform_method=config['data_transformer'], scaler=None)
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
                
                val_dataset = DataSetBuilder(kihadataset_val['x'], kihadataset_val['y'], kihadataset_val['labels'],
                                            transform_method=config['data_transformer'], scaler=train_dataset.scaler)
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=False)
               
                for epoch in range(num_epochs):
                    transformer_model.train()
                    epoch_loss = 0
                    for x_batch, y_batch in train_dataloader:
                        x_batch = x_batch.float()  # Convert to float32
                        y_batch = y_batch.float()  # Convert to float32
                        optimizer.zero_grad()
                        y_pred = transformer_model(x_batch)
                        loss = criterion(y_pred, y_batch)
                        loss.backward()
                        optimizer.step()
                    epoch_loss /= len(train_dataloader)
                    train_losses.append(epoch_loss)
                    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

                    if (epoch + 1) % 10 == 0:
                        transformer_model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for x_batch, y_batch in val_dataloader:
                                x_batch = x_batch.float()
                                y_batch = y_batch.float()
                                y_pred = transformer_model(x_batch)
                                loss = criterion(y_pred, y_batch)
                                val_loss += loss.item()
                        val_loss /= len(val_dataloader)
                        val_losses.append(val_loss)
                        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')

                        # Save checkpoint every 10 epochs
                        checkpoint_path = os.path.join('./caches/trained_model/', f'checkpoint_epoch_{epoch+1}.pt')
                        torch.save(transformer_model.state_dict(), checkpoint_path)
                        print(f"Checkpoint saved at {checkpoint_path}")

                #Plot the loss
                plt.figure()
                plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss Over Epochs')
                plt.legend()
                plt.savefig('training_loss_plot.png')
                plt.show()
                print("Training loss plot saved as training_loss_plot.png")
                # Evaluate the transformer model
                
                transformer_model.eval()
                test_dataset = DataSetBuilder(kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'],
                                            transform_method=config['data_transformer'], scaler=train_dataset.scaler)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)
                y_pred_list = []
                y_true_list = []

                with open('test_dataset.pkl', 'wb') as f:
                        pickle.dump({
                            'x': test_dataset.x.numpy(),
                            'y': test_dataset.y.numpy(),
                            'labels': test_dataset.labels
                        }, f)
                print("Train dataset saved to test_dataset.pkl")

                with torch.no_grad():
                    for x_batch, y_batch in test_dataloader:
                        x_batch = x_batch.float()  
                        y_batch = y_batch.float() 
                        y_pred = transformer_model(x_batch)
                        y_pred_list.append(y_pred)
                        y_true_list.append(y_batch)

                y_pred = torch.cat(y_pred_list, dim=0).numpy()
                y_true = torch.cat(y_true_list, dim=0).numpy()

                if save_model:
                        torch.save(transformer_model.state_dict(), os.path.join('./caches/trained_model/', 'final_transformer_model(APIA).pt'))

        else:
            # Classification
            train_dataset = DataSetBuilder(kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'],
                                           transform_method=config['data_transformer'], scaler=None)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
            test_dataset = DataSetBuilder(kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'],
                                          transform_method=config['data_transformer'], scaler=train_dataset.scaler)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)

            config['model_train_activity'], config['model_test_activity'] = get_model_name_from_activites(config['train_activity'],
                                                                                                          config['test_activity'])
            model_file = config['model_name'] + '_' + "".join(config['model_train_activity']) + \
                         '_' + "".join(config['model_test_activity']) + '2.pt'
            training_handler = Train(config, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
            model = training_handler.run_training()
            if save_model:
                torch.save(model, os.path.join('./caches/trained_model/', model_file))

            # Testing
            test_handler = Test()
            y_pred, y_true, loss = test_handler.run_testing(config, model, test_dataloader=test_dataloader)
            y_true = y_true.detach().cpu().clone().numpy()
            y_pred = y_pred.detach().cpu().clone().numpy()
            # Evaluation
            if individual_plot:
                for subject in config['test_subjects']:
                    subject_index = kihadataset_test['labels'][kihadataset_test['labels']['subject'] == subject].index.values
                    wandb_plotly_true_pred(y_true[subject_index], y_pred[subject_index], config['selected_opensim_labels'], str('test_'+ subject))
                    Evaluation(config=config, y_pred=y_pred[subject_index], y_true=y_true[subject_index], val_or_test=subject)
        # Evaluation
        results = {}
        for activity in config['test_activity']:
            activity_to_evaluate = activity
            activity_index = get_activity_index_test(kihadataset_test['labels'], activity_to_evaluate)
            wandb_plotly_true_pred(y_true[activity_index], y_pred[activity_index], config['selected_opensim_labels'], 'test_' + activity_to_evaluate)
            Evaluation(config=config, y_pred=y_pred[activity_index], y_true=y_true[activity_index], val_or_test='all_' + activity_to_evaluate)
            results[activity] = {
                'y_true': y_true[activity_index],
                'y_pred': y_pred[activity_index],
                'labels': kihadataset_test['labels'][activity_index]
            }
        with open('results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("Results saved to results.pkl")


if __name__ == '__main__':
    run_main()

