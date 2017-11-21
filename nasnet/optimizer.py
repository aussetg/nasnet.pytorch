from torch.optim.optimizer import Optimizer, required
import torch
import math


class PowersignCD(Optimizer):
    def __init__(self, params, steps, lr=required, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PowersignCD, self).__init__(params, defaults)
        self.t = 0
        self.T = steps

    def __setstate__(self, state):
        super(PowersignCD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data

                if weight_decay != 0:
                    g.add_(weight_decay, p.data)

                param_state = self.state[p]
                if 'exp_avg' not in param_state:
                    m = param_state['exp_avg'] = g.clone()
                else:
                    m = param_state['exp_avg']
                    m.mul_(momentum).add_(1 - momentum, g)

                w = torch.sign(g).mul(torch.sign(m))
                w.mul(.5*(1+math.cos(math.pi*self.t/self.T)))
                w.exp_()
                p.data.addcmul_(-.5 * (1 + math.cos(math.pi * self.t / self.T)) * group['lr'], w, g)

        self.t += 1

        return loss
