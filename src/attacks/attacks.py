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
    
    def single_attack(self, inputs, target):
        self.setup_attack(inputs, target)
        self.reset_grads(inputs)
        original_outputs = self.predict(inputs, only_output=True)
        self.compute_grads(original_outputs, target)
        input_grad_sign = self.gradient_sign(inputs)
        perturbed_images = self.perturb_image(inputs, input_grad_sign)

        return perturbed_images

class ProjectedGradientDescent(nn.Module):

    def __init__(self, model, loss_fn, iterations = 100, device = None, epsilon = 0.25, return_logits = False, norm = 'inf'):
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

    def random_start(self, ball_center):
        if self.norm == 'l2':
            rand_init = torch.randn_like(ball_center)
            unit_init = F.normalize(rand_init.view(rand_init.size(0), -1)).view(rand_init.size())
            number_elements = torch.numel(ball_center)
            r = (torch.rand(rand_init.size(0)) ** (1.0 / number_elements)) * self.epsilon
            r = r[(...,) + (None,) * (r.dim() - 1)]
            move_away = r * unit_init
            return ball_center + move_away
        elif self.norm == 'inf':
            move_away = torch.rand_like(ball_center) * self.epsilon * 2 - self.epsilon
            return ball_center + move_away

    def forward(self, inputs, target, iterations = None):

        self.model.train(False)
        inputs.requires_grad = True
        inputs, target = inputs.to(self.device), target.to(self.device)

        self.model.zero_grad()
        if inputs.grad:
            inputs.grad.zero_()

        perturbed_images = inputs.copy()

        outputs = self.model(perturbed_images)

        # Original prediction
        with torch.no_grad():
            original_preds = outputs.argmax(1)

        if iterations:
            num_iterations = iterations
        else:
            num_iterations = self.iterations

        for it in range(num_iterations):
            perturbed_image = self.fgsm.single_attack(perturbed_image, target)
            """
            loss = self.loss_fn(outputs, target)
            loss.backward()
            
            input_grad = inputs.grad

            update_grad = input_grad.sign()
            
            perturbed_images = perturbed_images + self.epsilon * update_grad

            perturbed_images

            self.model.zero_grad()
            inputs.grad.zero_()

            outputs = self.model(perturbed_images)
            """


        # outputs = self.model(perturbed_images)
        # New prediction
        with torch.no_grad():
            new_preds = outputs.argmax(1)
        
        if self.return_logits:
            return perturbed_images, original_preds, new_preds, outputs, new_outputs
        
        return perturbed_images, original_preds, new_preds
        








