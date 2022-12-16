class CustomScheduler():
    def __init__(self, optimizer, epoch = 1):
        self.epoch = epoch
        self.optimizer = optimizer
        self.init_lr = optimizer.defaults['lr']
        
    def step(self):
        self.epoch += 1
        if self.epoch <= 5:
            for g in self.optimizer.param_groups:
                g['lr'] = self.init_lr * self.epoch
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5
                
