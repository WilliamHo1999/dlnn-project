import torch
import torch.nn as nn

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

    def forward(self, inputs, target):

        self.model.train(False)
        inputs.requires_grad = True
        inputs, target = inputs.to(self.device), target.to(self.device)

        self.model.zero_grad()
        if inputs.grad:
            inputs.grad.zero_()        

        outputs = self.model(inputs)

        # Original prediction
        with torch.no_grad():
            original_preds = outputs.argmax(1)
        loss = self.loss_fn(outputs, target)
        loss.backward()

        input_grad = inputs.grad
        input_grad_sign = input_grad.sign()
        
        perturbed_images = inputs + self.epsilon * input_grad_sign

        new_outputs = self.model(perturbed_images)
        
        # New prediction
        with torch.no_grad():
            new_preds = new_outputs.argmax(1)
        
        if self.return_logits:
            return original_preds, new_preds, outputs, new_outputs
        
        return original_preds, new_preds
        

        


            
        








