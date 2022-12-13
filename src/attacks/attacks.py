import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
    def forward(self, inputs, target, iterations = None):

        self.model.train(False)
        inputs.requires_grad = True
        inputs, target = inputs.to(self.device), target.to(self.device)

        self.model.zero_grad()
        if inputs.grad is not None:
            inputs.grad.zero_()

        perturbed_images = inputs.clone().detach()

        # Original prediction
        with torch.no_grad():
            outputs = self.model(perturbed_images)
            original_preds = outputs.argmax(1)

        if iterations:
            num_iterations = iterations
        else:
            num_iterations = self.iterations        
    
        # perturbed_images = self.random_start(perturbed_images)
        
        # iterate
        for it in range(num_iterations):
            # Fast gradient sign attack
            perturbed_images = self.fgsm.single_attack(perturbed_images, target)
            # Clip to keep sample inside the ball
            perturbed_images = self.clip(inputs, perturbed_images)
            # Remove computational graph to allow for gradients
            perturbed_images = perturbed_images.detach()

        with torch.no_grad():
            new_outputs = self.model(perturbed_images)
            new_preds = new_outputs.argmax(1)
        
        if self.return_logits:
            return perturbed_images, original_preds, new_preds, outputs, new_outputs
        
        return perturbed_images, original_preds, new_preds






