import os
import numpy as np
import PIL.Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt


class ImageDisplayerGradCam():

    def __init__(self, model, grad_cam, classes, reshape = transforms.Resize((256,256)), multi_label = True, image_dir = '', pdf = False, save_figs = False):


        self.model = model
        self.grad_cam = grad_cam

        self.multi_label = multi_label

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.classes = classes

        self.image_dir = image_dir

        self.pdf = pdf
        
        # True = Display labels, False = Display Predictions
        self.display_labels_or_predictions = True
    
        self.save_figs = save_figs


        # Reshape image, as 
        self.reshape = reshape


    def display_images(self, data, target_class = None, display_labels_or_predictions = True, return_heatmap = False):

        return self._display_images(data, target_class, display_labels_or_predictions, return_heatmap)


    def _display_images(self, data, target_class , display_labels_or_predictions, return_heatmap):
        
        self.display_labels_or_predictions = display_labels_or_predictions

        self.model.eval()

        image = data['image'].to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        labels = data['label']
        image_file_name = data['filename']

        output = self.model(image)

        if self.multi_label:
            output = torch.sigmoid(output)
            output = output > 0.5
        else:
            output = torch.argmax(output, axis = 1)

        # Retreive predictions, to know what the model predicted
        if target_class is not None:
            return self._display_target_class(image, output, labels, target_class, image_file_name, return_heatmap)
        elif self.multi_label:
            return self._display_multi_label(image, output, labels, image_file_name, return_heatmap)
        else:
            return self._display_single_label(image, output, labels, image_file_name, return_heatmap)

    def _display_multi_label(self, image, prediction, labels, image_file_name, return_heatmap):
        """
        Generate sailency maps for each of the 
        """
        if prediction.nonzero().nelement() == 0:
            predictions_str_list = ["None"]
        else:
            pred_ind = prediction.squeeze(0).cpu().numpy()
            predictions_str_list = [f"{self.classes[ind]} ({ind})" for ind, pred in enumerate(pred_ind) if pred == 1]

        true_classes_str_list = [f"{self.classes[ind]} ({ind})" for ind, lab in enumerate(labels) if lab == 1]

        print("True classes:", ', '.join(true_classes_str_list))
        print("Predictions:",  ', '.join(predictions_str_list))

        if self.display_labels_or_predictions:
            classes_to_iterate = [ind for ind, lab in enumerate(labels) if lab == 1]
            print("Displaying true labels")
            target_classes_str_list = true_classes_str_list
        else:
            classes_to_iterate = [ind for ind, pred in enumerate(pred_ind) if pred == 1]
            print("Displaying predictions")
            target_classes_str_list = predictions_str_list

        for target_nr, target in enumerate(classes_to_iterate):

            heatmap, _ = self.grad_cam(image, target)

            target_class_str = target_classes_str_list[target_nr]

            return self._show_image_with_heapmap(heatmap, target, true_classes_str_list, target_class_str, image_file_name, return_heatmap=return_heatmap)


    def _display_single_label(self, image, prediction, label, image_file_name, return_heatmap):

        prediction = prediction.item()
        try:
            label = label.item()
        except:
            pass
        
        print("True class:", self.classes[label], f"{label}")
        print("Predictions:",  self.classes[prediction], f"{prediction}")

        if self.display_labels_or_predictions:
            target_class = label
            print("Displaying true labels")
        else:
            target_class = prediction
            print("Displaying predictions")

        heatmap, _ = self.grad_cam(image, target_class)

        print("_display_target_class", image.size())

        true_class_str = [f"{self.classes[label]} ({label})"]
        target_class_str = f"{self.classes[target_class]} ({target_class})"

         ### edits 2022-12-15
        if image_file_name is not None:
            return self._show_image_with_heapmap(heatmap, target_class, true_class_str, target_class_str, image_file_name, return_heatmap=return_heatmap)
        else:
            return self._show_image_with_heapmap(heatmap, target_class, true_class_str, target_class_str, image_file_name, image=image, return_heatmap=return_heatmap)


    def _display_target_class(self, image, prediction, labels, target_class, image_file_name, return_heatmap):
        
        if self.multi_label:
            true_classes_str_list = [f"{self.classes[ind]} ({ind})" for ind, lab in enumerate(labels) if lab == 1]
            print('True classes:', '\n'.join(true_classes_str_list))
        else:
            try:
                true_cls = labels[0].item()
            except Exception:
                true_cls = labels
            true_classes_str_list = [f"{self.classes[true_cls]} ({true_cls})"]
            print("True Class:", true_cls)
        
        target_class_str =  f"{self.classes[target_class]} ({target_class})"
        print("Target Class:", target_class_str)

        heatmap, _ = self.grad_cam(image, target_class)

        ### edits 2022-12-15
        if image_file_name is not None:
            return self._show_image_with_heapmap(heatmap, target_class, true_classes_str_list, target_class_str, image_file_name, return_heatmap=return_heatmap)
        else:
            return self._show_image_with_heapmap(heatmap, target_class, true_classes_str_list, target_class_str, image_file_name, image=image, return_heatmap=return_heatmap)
        

    def _show_image_with_heapmap(self, heatmap, target_class, true_classes_str_list, target_class_str, image_file_name, image=None, return_heatmap = False):
        fig, ax = plt.subplots(1, 1,)

        if isinstance(image_file_name, list):
            image_file_name = image_file_name[0]

        ### edits 2022-12-15
        #print("heatmap", heatmap)
        if image_file_name is not None:
            image = self.reshape((PIL.Image.open(image_file_name)))
        elif image is not None:
            image_255 = (image[0].numpy()*255).astype(np.uint8)
            print(image[0].size())
            image = (PIL.Image.fromarray(image_255.transpose(1,2,0), mode="RGB"))
        
        ax.axis('off')
        ax.imshow(image)
        
        true_classes  = ', '.join(true_classes_str_list)
        ax.set_title(f'True class(es): {true_classes}. \nTarget class: {target_class_str}')

        cmap = plt.get_cmap('jet')
        cmap._init()
        cmap._lut[:,-1] = np.linspace(0, 0.8, 255+4)
        w, h = image.size
        y, x = np.mgrid[0:h, 0:w]
        cb = ax.contourf(x, y, heatmap, 15, cmap=cmap)
        plt.colorbar(cb)
        
        if image_file_name is not None: 
            if os.name == 'nt':
                file_name = image_file_name.split('.')[0].split('\\')[-1]
            else:
                file_name = image_file_name.split('.')[0].split('/')[-1]

            if self.pdf:
                print(f'{self.image_dir}/pdf/{file_name}_heatmap_{target_class}.pdf')
                if self.save_figs:
                    plt.savefig(f'{self.image_dir}/{file_name}_heatmap_{target_class}.pdf')
            else:
                print(f'{self.image_dir}/{file_name}_heatmap_{target_class}.png')
                if self.save_figs:
                    plt.savefig(f'{self.image_dir}/{file_name}_heatmap_{target_class}.png')
        plt.show()

        if return_heatmap:
            return heatmap