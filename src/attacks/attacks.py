import torch
import torch.nn as nn
import torch.nn.functional as F
#import deepfool as deepfool
from . import deepfool

class FastGradientSign(nn.Module):
    
    def __init__(self, model, loss_fn, device = None, epsilon = 0.25, return_logits = False):
        super().__init__()

        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn
        self.return_logits = return_logits

    def setup_attack(self, inputs, target):
        self.model.train(False)
        inputs.requires_grad = True
        inputs, target = inputs.to(self.device), target.to(self.device)

    def reset_grads(self, inputs = None):
        self.model.zero_grad()
        if inputs.grad is not None:
            inputs.grad.zero_()    

    def predict(self, inputs, only_output = False):
        outputs = self.model(inputs)
        if only_output:
            return outputs
        with torch.no_grad():
            prediction = outputs.argmax(1)
        return outputs, prediction
    
    def compute_grads(self, outputs, target):
        loss = self.loss_fn(outputs, target)
        loss.backward()

    def gradient_sign(self, inputs):
        input_grad = inputs.grad
        input_grad_sign = input_grad.sign()
        return input_grad_sign

    def perturb_image(self, inputs, input_grad_sign):
        perturbed_images = inputs + self.epsilon * input_grad_sign
        return perturbed_images

    def forward(self, inputs, target):
        
        self.setup_attack(inputs, target)
        self.reset_grads(inputs)
        original_outputs, original_prediction = self.predict(inputs)
        self.compute_grads(original_outputs, target)

        input_grad_sign = self.gradient_sign(inputs)
        perturbed_images = self.perturb_image(inputs, input_grad_sign)

        new_outputs, new_preds = self.predict(perturbed_images)
    
        if self.return_logits:
            return perturbed_images, original_prediction, new_preds, original_outputs, new_outputs
        
        return perturbed_images, original_prediction, new_preds
    
    def single_attack(self, inputs, target, setup = False):
        #if setup:
        self.setup_attack(inputs, target)
        self.reset_grads(inputs)
        original_outputs = self.predict(inputs, only_output=True)
        self.compute_grads(original_outputs, target)
        input_grad_sign = self.gradient_sign(inputs)
        perturbed_images = self.perturb_image(inputs, input_grad_sign)

        return perturbed_images

class ProjectedGradientDescent(nn.Module):

    def __init__(self, model, loss_fn, iterations = 20, device = None, epsilon = 0.25, return_logits = False, norm = 'inf'):
        super().__init__()
        """
        args:
            norm: "l2" or "inf"
        """

        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.return_logits = return_logits

        self.norm = norm.lower()

        self.iterations = iterations

        self.fgsm = FastGradientSign(model, loss_fn, device, epsilon, return_logits)

    def clip(self, original_images, perturbed_images):
        diff = perturbed_images - original_images
        if self.norm == 'inf':
            return original_images + torch.clamp(diff, -self.epsilon, self.epsilon)
        elif self.norm == 'l2':
            return original_images + torch.renorm(diff, 2,0, self.epsilon)

    def random_start(self, ball_center):
        if self.norm == 'l2':
            rand_init = torch.randn_like(ball_center)
            unit_init = F.normalize(rand_init.view(rand_init.size(0), -1)).view(rand_init.size())
            number_elements = torch.numel(ball_center)
            r = (torch.rand(rand_init.size(0)) ** (1.0 / number_elements)) * self.epsilon
            r = r[(...,) + (None,) * (r.dim() - 1)]
            move_away = r * unit_init
            return ret
        elif self.norm == 'inf':
            move_away = torch.rand_like(ball_center) * self.epsilon * 2 - self.epsilon
            ret = ball_center + move_away
            ret.requires_grad = True
            return ret

    def forward(self, inputs, target, iterations = None, compute_original_prediction = True, compute_new_preds = True, random_start = False):

        self.model.train(False)
        inputs.requires_grad = True
        inputs, target = inputs.to(self.device), target.to(self.device)

        self.model.zero_grad()
        if inputs.grad is not None:
            inputs.grad.zero_()

        perturbed_images = inputs.clone().detach()

        if compute_original_prediction:
            # Original prediction
            with torch.no_grad():
                outputs = self.model(perturbed_images)
                original_preds = outputs.argmax(1)

        if iterations:
            num_iterations = iterations
        else:
            num_iterations = self.iterations        
        
        if random_start:
            perturbed_images = self.random_start(perturbed_images)
        
        # iterate
        for it in range(num_iterations):
            # Fast gradient sign attack
            perturbed_images = self.fgsm.single_attack(perturbed_images, target)
            # Clip to keep sample inside the ball
            perturbed_images = self.clip(inputs, perturbed_images)
            # Remove computational graph to allow for gradients
            perturbed_images = perturbed_images.detach()

        if compute_new_preds:
            with torch.no_grad():
                new_outputs = self.model(perturbed_images)
                new_preds = new_outputs.argmax(1)
        
        if self.return_logits:
            if compute_original_prediction and compute_new_preds:
                return perturbed_images, original_preds, new_preds, outputs, new_outputs
            elif compute_original_prediction:
                return perturbed_images, original_preds, outputs
            elif compute_new_preds:
                return perturbed_images, new_preds, new_outputs
            else:
                return perturbed_images

        if compute_original_prediction and compute_new_preds:
            return perturbed_images, original_preds, new_preds
        elif compute_original_prediction:
            return perturbed_images, original_preds
        elif compute_new_preds:
            return perturbed_images, new_preds
        else:
            return perturbed_images

# implemented according to the arxiv paper
class UniversalPerturbation(nn.Module):
    def __init__(self, model, loss_fn, device = None, epsilon = 0.25, delta=0.05, return_logits = False):
        super().__init__()

        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model
        self.epsilon = epsilon
        self.delta = delta
        self.loss_fn = loss_fn
        self.return_logits = return_logits

    def predict(self, outputs): # get class labels by finding largest value among class predictions
        return torch.squeeze(outputs.max(-1,keepdim=True)[1])  # assumes output of 1 sample

    def forward_predict(self, inputs): # combine model() and predict()
        outputs = self.model(inputs)
        return self.predict(outputs)

    # TODO: ASK WILLIAM
    def _project_perturbation(self,perturbation,epsilon,l_norm_p_val="inf"):
        if l_norm_p_val == "2":
            return torch.renorm(perturbation, 2, 0, epsilon)
        elif l_norm_p_val == "inf":
            return torch.clamp(perturbation, -epsilon, epsilon)

    def _find_single_point_perturbation(self, point):
        pass

    def _compute_fool_rate(self, inputs, targets):
        outputs = self.model(inputs)
        predictions = self.predict(outputs)
        print("predictions: ", predictions.size(), "\n")
        print("targets", targets.size(), "\n")
        no_incorrect_predictions = torch.numel(predictions[predictions != targets])
        fool_rate = float(no_incorrect_predictions)/torch.numel(targets)
        print("fool_rate", fool_rate)
        return fool_rate

    # TODO: problems with l2 norm for projection -> hinders convergence
    def compute_perturbation(self, inputs, targets, max_iter = 99999, max_iter_df = 50):
        delta = 0.05  # threshold for fool rate
        epsilon = 1 # size of l-norm ball to project into # determines whether while loop converges!

        # intialize
        iter = 0
        v = torch.zeros_like(inputs[0]) # perturbation
        inputs_perturbed = inputs.clone()

        print("inputs", inputs_perturbed.size())
        # main loop: as long as error is too low (too little images are incorrectly classified)
        while self._compute_fool_rate(inputs_perturbed, targets) < 1-delta and iter < max_iter:
            print(iter)
            for i in range(inputs.size()[0]): # iterate over datapoints (assumed all given in one data matrix)
            #for datapoint in inputs.T: # each column is datapoint
                datapoint = torch.unsqueeze(inputs[i,:,:,:],0)
                #print("datapoint", datapoint.size())
                #print("self.forward_predict(datapoint + v)", self.forward_predict(datapoint + v).size())
                #print("self.forward_predict(datapoint)", self.forward_predict(datapoint).size())
                if self.forward_predict(datapoint + v) == self.forward_predict(datapoint):
                    # find perturbation that misclassfies this sample
                    no_classes = list(self.model.children())[-1].out_features
                    #print("no_classes", no_classes)
                    v_sample, _, _, _, _ = deepfool.deepfool(torch.squeeze(datapoint),self.model,
                        num_classes=no_classes, max_iter=max_iter_df, overshoot=0.02)
                # update perturbation by 
                # 1) adding the new perturbation to the previous, 
                v = v + v_sample
                ###print("v", v.expand(inputs.size()).size(), v[:10])
                # 2) projecting back into the l-constraint space
                v = self._project_perturbation(v, epsilon, "inf")
                ###print("v", v.expand(inputs.size()).size(), v[:10])
            # update inputs !!!
            inputs_perturbed = inputs + v.expand(inputs.size())
            iter += 1
        return v
                