import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import os
import sys
import time

class Trainer():

    def __init__(self, model, loss_fn, classes, training_params, device = 'cuda', num_epochs = 5, model_name=None, save_model = True, model_dir = 'models', multi_label = False, print_frequency = 100, adversarial_training = False, adversarial_attack = None, custom_scheduler = False, continue_dict = None, mnist = False):
        
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
        self.epoch_mape = {'train': [], 'val': []}
        self.epoch_ap = {'train': np.zeros((1,self.num_classes)), 'val': np.zeros((1,self.num_classes))}
        
        self.confusion_matrix = {
            "train": np.zeros(shape=(self.num_classes , self.num_classes , self.num_epochs)),
            "val": np.zeros(shape=(self.num_classes , self.num_classes , self.num_epochs)),
        }
        
        if continue_dict:
            self.best_model = model
            self.best_measure = continue_dict["best_measure"]
            self.train_measure_at_best = continue_dict["train_measure_at_best"]
            self.train_loss_at_best = continue_dict["train_loss_at_best"]
            self.best_eval_loss = continue_dict[ "best_eval_loss"]
            self.best_loss = continue_dict[ "best_eval_loss"]
            self.best_epoch = continue_dict["best_epoch"]
            self.latest_epoch = continue_dict["latest_epoch"]
            self.start_epoch = continue_dict["latest_epoch"] + 1
        else:
            self.best_model = None
            self.best_measure = 0
            self.train_measure_at_best = 0
            self.train_loss_at_best = 0
            self.best_eval_loss = 0
            self.best_epoch = 0
            self.latest_epoch = 0
            self.start_epoch = 0
            self.best_loss = 0
        
        self.optimizer = training_params['optimizer']
        self.scheduler = training_params['scheduler']

        self.stop_iteration = 0
        
        self.multi_label = multi_label
        
        self.print_frequency = print_frequency
        
        self.longest_class_name = max([len(i) for i in self.class_names])
        
        self.adversarial_training = adversarial_training
        self.adversarial_attack = adversarial_attack
        self.custom_scheduler = custom_scheduler
        self.mnist = mnist


    def train(self):

        for current_epoch in range(self.start_epoch, self.num_epochs):
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
                self.best_eval_loss = self.epoch_loss['val'][-1]
                self.best_epoch = current_epoch

                if self.save_model:
                    assert self.best_model is not None
                    if not os.path.isdir(self.model_dir):
                        os.makedirs(self.model_dir)
                        
                    model_name_pt = self.model_name+'.pt'
                    PATH = os.path.join(self.model_dir, model_name_pt)
                    self.best_model.to('cpu')
                    if self.custom_scheduler:
                        torch.save({
                            'epoch': current_epoch+1,
                            'model_state_dict': self.best_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }, PATH)
                    else:
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
                if self.stop_iteration == 11:
                    return

            # print progress
            if self.multi_label:
                print("Current Epoch:",current_epoch)
                print("Eval  Model: ", self.model_name, ". MAPE: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
                print("Train Model: ", self.model_name, ". MAPE", self.epoch_mape['train'][-1], "Avg loss:",  self.epoch_loss['train'][-1])
                print("Current MAPE: ",self.best_measure, self.best_eval_loss, "at epoch",self.best_epoch)
                for ii in range(len(self.classes)):
                    ap = self.epoch_ap['val'][-1, ii]
                    print(
                        f'AP of {str(self.class_names[ii]).ljust(self.longest_class_name+2)}: {ap}'
                   #     f'{ap*100:.01f}'
                    )
            else:
                print("Current Epoch:",current_epoch)
                print("Eval  Model: ", self.model_name, ". Acc: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
                print("Train Model: ", self.model_name, ". Acc", self.epoch_acc['train'][-1], "Avg loss:",  self.epoch_loss['train'][-1])
                print("Current acc: ",self.best_measure, self.best_eval_loss, "at epoch",self.best_epoch)
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
            print("Eval Model: ", self.model_name, ". MAPE: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
            for ii in range(len(self.classes)):
                ap = self.epoch_ap['val'][-1, ii]
                print(
                    f'AP of {str(self.class_names[ii]).ljust(self.longest_class_name+2)}: {ap}'
#                    f'{ap*100:.01f}'
                )
        else:
            print("Current Epoch:",0)
            print("Eval Model: ", self.model_name, ". Acc: ", eval_measure, ". Avg loss:",  self.epoch_loss['val'][-1])
            for ii in range(len(self.classes)):
                acc = self.confusion_matrix['val'][ii, ii, current_epoch] / np.sum(
                    self.confusion_matrix['val'][ii, :, current_epoch], )
                print(
                        f'Accuracy of {str(self.class_names[ii]).ljust(70)}:  {acc}'
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
                if self.mnist:
                    inputs = data[0].to(self.data_device)
                    labels = data[1].to(self.data_device)
                else:
                    inputs = data.get('image').to(self.data_device)
                    labels = data.get('label').to(self.data_device)
                
                if self.adversarial_training:
                    inputs = self.adversarial_attack(inputs, labels, random_start = True, compute_original_prediction = False, compute_new_preds = False)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

            else:
                if self.mnist:
                    inputs = data[0].to(self.data_device)
                    labels = data[1].to(self.data_device)
                else:
                    inputs = data.get('image').to(self.data_device)
                    labels = data.get('label').to(self.data_device)
                    
                if self.adversarial_training:
                    inputs = self.adversarial_attack(inputs, labels, random_start = True, compute_original_prediction = False, compute_new_preds = False)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels) 

            if is_training:
                loss.backward()
                self.optimizer.step()

                
            losses.append(loss.item())
            # Multiple Labels per image
            if self.multi_label:
                cpuout = outputs.detach().to('cpu')
                pred_scores = cpuout.numpy() 
                concat_pred = np.append(concat_pred, pred_scores, axis = 0)
                concat_labels = np.append(concat_labels, labels.cpu().numpy(), axis = 0)
            # Single label per image
            else:
                preds = torch.argmax(outputs, dim = 1)
                #print("Unique preds:",torch.unique(preds).shape)
                #print("Unique Labels:",torch.unique(labels).shape)
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