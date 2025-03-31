# Implementation for computing exponential moving averages (EMA) of model parameters

## this may be for stabilitization
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates=99999):
        ## computes an exponential moving average of the model, for every time the model's called
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        ## everytime its called the decay decreases ?, so older model matters less and less.
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                ## exp movnig average of previous passes and current passes of model parameters.
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        ## reassigns model paramters to non exponential moving average
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]