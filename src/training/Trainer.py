import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import os
import sys
import time

class Trainer():

    def __init__(self, model, loss_fn, classes, training_params, device = 'cuda', num_epochs = 5, model_name=None, save_model = True, model_dir = 'models', multi_label = False, print_frequency = 100):
        
        self.model = model
        self.device = device
        self.data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.class_names = classes
        self.num_classes = len(classes)
        self.classes = [i for i in range(self.num_classes)]
        
        self.save_model = save_model
        self.model_dir = model_dir

        self.loss_fn = loss_fn
        self.data_loader = training_params['dataloaders']
        self.num_epochs = num_epochs

        self.epoch_loss = {'train': [], 'val': []}
        self.epoch_acc = {'train': [], 'val': []}
        
        self.confusion_matrix = {
            "train": np.zeros(shape=(self.num_classes , self.num_classes , self.num_epochs)),
            "val": np.zeros(shape=(self.num_classes , self.num_classes , self.num_epochs)),
        }
        
        self.best_model = None
        self.best_measure = 0
        self.train_measure_at_best = 0
        self.train_loss_at_best = 0
        self.best_eval_loss = 0
        self.best_epoch = 0
        self.latest_epoch = 0
        
        self.optimizer = training_params['optimizer']
        self.scheduler = training_params['scheduler']

        self.stop_iteration = 0
        
        self.multi_label = multi_label
        
        self.print_frequency = print_frequency
        
        self.longest_class_name = max([len(i) for i in self.class_names])


    def train(self):

        for current_epoch in range(self.num_epochs):
            self.latest_epoch = current_epoch
            
            print("Current Epoch:", current_epoch)
            print("Train Loop:")
            tic = time.time()
            self.model.train()
            train_dict = self.run_epoch(
                self.model,
                current_epoch,
                is_training=True,
                mode='train',
            )
            print(f"Train loss: {train_dict['loss']}. Train measure: {train_dict['measure']}")
            print(f"Train loop took {time.time()-tic}")

            
            tic = time.time()
            print("Val Loop:")
            self.model.eval()
            eval_dict = self.run_epoch(
                self.model,
                current_epoch,
                is_training=False,
                mode='val',
            )
            
            print(f"Validation loss: {eval_dict['loss']}. Val measure: {eval_dict['measure']}")
            print(f"Validation loop took {time.time()-tic}")

            self.scheduler.step()

            eval_measure = eval_dict['measure']

            if eval_measure > self.best_measure:
                self.stop_iteration = 0
                self.best_measure = eval_measure
                self.train_loss_at_best = train_dict['loss']
                self.train_measure_at_best = train_dict['measure']
                self.best_model = self.model
                self.best_loss = self.epoch_loss['val'][-1]
                self.best_epoch = current_epoch

                if self.save_model:
                    assert self.best_model is not None
                    if not os.path.isdir(self.model_dir):
                        os.makedirs(self.model_dir)
                        
                    model_name_pt = self.model_name+'.pt'
                    PATH = os.path.join(self.model_dir, model_name_pt)
                    self.best_model.to('cpu')
                    torch.save({
                        'epoch': current_epoch+1,
                        'model_state_dict': self.best_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }, PATH)
                    self.best_model.to(self.device)
                    self.model_path = PATH
                    
            else:
                self.stop_iteration += 1
                
                # No improvement in 5 epochs
                if self.stop_iteration == 6:
                    return

            # print progress
            if self.multi_label:
                print("Current Epoch:",current_epoch)
                print("Eval  Model: ", self.model_name, ". MAPE: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
                print("Train Model: ", self.model_name, ". MAPE", self.epoch_mape['train'][-1], "Avg loss:",  self.epoch_loss['train'][-1])
                print("Current MAPE: ",self.best_measure, self.best_loss, "at epoch",self.best_epoch)
                for ii in range(len(self.classes)):
                    ap = self.epoch_ap['val'][-1, ii]
                    print(
                        f'AP of {str(self.class_names[ii]).ljust(self.longest_class_name+2)}: {ap}'
                        f'{ap*100:.01f}%'
                    )
            else:
                print("Current Epoch:",current_epoch)
                print("Eval  Model: ", self.model_name, ". Acc: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
                print("Train Model: ", self.model_name, ". Acc", self.epoch_acc['train'][-1], "Avg loss:",  self.epoch_loss['train'][-1])
                print("Current acc: ",self.best_measure, self.best_loss, "at epoch",self.best_epoch)
                for ii in range(len(self.classes)):
                    acc = self.confusion_matrix['val'][ii, ii, current_epoch] / np.sum(
                        self.confusion_matrix['val'][ii, :, current_epoch], )
                    print(
                        f'Accuracy of {str(self.class_names[ii]).ljust(70)}:  {acc}'
                    )


    def run_validation_loop_only(self):
        current_epoch = 0
        
        tic = time.time()
        print("Val Loop:")
        self.model.eval()
        eval_dict = self.run_epoch(
            self.model,
            0,
            is_training=False,
            mode='val',
        )
        
        eval_measure = eval_dict['measure']

        print(f"Validation loss: {eval_dict['loss']}. Val measure: {eval_dict['measure']}")
        print(f"Validation loop took {time.time()-tic}")
        
# print progress
        if self.multi_label:
            print("Current Epoch:",0)
            print("Eval  Model: ", self.model_name, ". MAPE: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
            for ii in range(len(self.classes)):
                ap = self.epoch_ap['val'][-1, ii]
                print(
                    f'AP of {str(self.class_names[ii]).ljust(self.longest_class_name+2)}: {ap}'
                    f'{ap*100:.01f}%'
                )
        else:
            print("Current Epoch:",0)
            print("Eval  Model: ", self.model_name, ". Acc: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
            for ii in range(len(self.classes)):
                acc = self.confusion_matrix['val'][ii, ii, current_epoch] / np.sum(
                    self.confusion_matrix['val'][ii, :, current_epoch], )
                print(
                    f'AP of {str(self.class_names[ii]).ljust(self.longest_class_name+2)}: {acc}'
                    f'{acc*100:.01f}%'
                )

    def run_epoch(self, model, current_epoch, is_training, mode):
        """
        Run epoch
        """
         
        if self.device == "cuda":
            torch.cuda.empty_cache()

        losses = []
        total_samples = len(self.data_loader[mode].dataset)
        # num_batches = len(self.data_loader[mode])
        # print_freq = round(num_batches / self.print_frequency)

        total_correct = 0

        # For Average Precision
        if self.multi_label:
            concat_pred = np.zeros((1,self.num_classes))
            concat_labels = np.zeros((1,self.num_classes))
            avgprecs = np.zeros(self.num_classes)
        # For accuracy
        else:
            confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes))

            
        for index, data in enumerate(self.data_loader[mode]):
            if is_training:
                inputs = data.get('image').to(self.data_device)
                labels = data.get('label').to(self.data_device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

            else:
                with torch.no_grad():
                    inputs = data['image'].to(self.data_device)       
                    labels = data['label'].to(self.data_device)

                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels) 

            if is_training:
                loss.backward()
                self.optimizer.step()

                
            losses.append(loss.item())
            # Multiple Labels per image
            if self.multi_label:
                cpuout= outputs.detach().to('cpu')
                pred_scores = cpuout.numpy() 
                concat_pred = np.append(concat_pred, pred_scores, axis = 0)
                concat_labels = np.append(concat_labels, labels.cpu().numpy(), axis = 0)
            # Single label per image
            else:
                preds = torch.argmax(outputs, dim = 1)
                total_correct += preds.eq(labels).cpu().sum().numpy()
                confusion_matrix += metrics.confusion_matrix(
                    labels.cpu().numpy(), preds.cpu().numpy(), labels=self.classes,
                )

            if index % self.print_frequency == 0:
                print(f'Batch: {index} of {len(self.data_loader[mode])}. Loss: {loss.item()}. Mean so far: {np.mean(losses)}. Mean of 100: {np.mean(losses[-100:])}')
        
        
        if self.multi_label:
            concat_pred = concat_pred[1:,:]
            concat_labels = concat_labels[1:,:]
            
            for c in range(self.num_classes):   
                avgprecs[c]=  metrics.average_precision_score(concat_labels[:,c], concat_pred[:,c])

            measure = np.mean(avgprecs)
            self.epoch_mape[mode].append(measure)
            self.epoch_ap[mode] = np.append(self.epoch_ap[mode], avgprecs.reshape(1,-1), axis = 0)
        else:
            measure = total_correct/total_samples
            self.epoch_acc[mode].append(measure)
            self.confusion_matrix[mode][:, :, current_epoch] = confusion_matrix
       
    
        loss = np.mean(losses)

        self.epoch_loss[mode].append(loss)

        return {"measure": measure, "loss": loss}