
import logging
import math
from functools import partial
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
log = logging.getLogger('flair')


class SGDW(Optimizer):
    'Implements stochastic gradient descent (optionally with momentum) with\n    weight decay from the paper `Fixing Weight Decay Regularization in Adam`_.\n\n    Nesterov momentum is based on the formula from\n    `On the importance of initialization and momentum in deep learning`__.\n\n    Args:\n        params (iterable): iterable of parameters to optimize or dicts defining\n            parameter groups\n        lr (float): learning rate\n        momentum (float, optional): momentum factor (default: 0)\n        weight_decay (float, optional): weight decay factor (default: 0)\n        dampening (float, optional): dampening for momentum (default: 0)\n        nesterov (bool, optional): enables Nesterov momentum (default: False)\n\n    .. _Fixing Weight Decay Regularization in Adam:\n        https://arxiv.org/abs/1711.05101\n\n    Example:\n        >>> optimizer = torch.optim.SGDW(model.parameters(), lr=0.1, momentum=0.9,\n                                         weight_decay=1e-5)\n        >>> optimizer.zero_grad()\n        >>> loss_fn(model(input), target).backward()\n        >>> optimizer.step()\n\n    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf\n\n    .. note::\n        The implementation of SGD with Momentum/Nesterov subtly differs from\n        Sutskever et. al. and implementations in some other frameworks.\n\n        Considering the specific case of Momentum, the update can be written as\n\n        .. math::\n                  v = \\rho * v + g \\\\\n                  p = p - lr * v\n\n        where p, g, v and :math:`\\rho` denote the parameters, gradient,\n        velocity, and momentum respectively.\n\n        This is in contrast to Sutskever et. al. and\n        other frameworks which employ an update of the form\n\n        .. math::\n             v = \\rho * v + lr * g \\\\\n             p = p - v\n\n        The Nesterov version is analogously modified.\n    '

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if ((lr is not required) and (lr < 0.0)):
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if (momentum < 0.0):
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if (weight_decay < 0.0):
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if (nesterov and ((momentum <= 0) or (dampening != 0))):
            raise ValueError(
                'Nesterov momentum requires a momentum and zero dampening')
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        'Performs a single optimization step.\n\n        Arguments:\n            closure (callable, optional): A closure that reevaluates the model\n                and returns the loss.\n        '
        loss = None
        if (closure is not None):
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if (p.grad is None):
                    continue
                d_p = p.grad.data
                if (momentum != 0):
                    param_state = self.state[p]
                    if ('momentum_buffer' not in param_state):
                        buf = param_state['momentum_buffer'] = torch.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_((1 - dampening), d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if (weight_decay != 0):
                    p.data.add_((- weight_decay), p.data)
                p.data.add_((- group['lr']), d_p)
        return loss


class AdamW(Optimizer):
    'Implements AdamW optimizer.\n\n    Adam has been proposed in `Adam\\: A Method for Stochastic Optimization`_.\n    AdamW uses the weight decay method from the paper\n    `Fixing Weight Decay Regularization in Adam`_.\n\n    Arguments:\n        params (iterable): iterable of parameters to optimize or dicts defining\n            parameter groups\n        lr (float, optional): learning rate (default: 1e-3)\n        betas (Tuple[float, float], optional): coefficients used for computing\n            running averages of gradient and its square (default: (0.9, 0.999))\n        eps (float, optional): term added to the denominator to improve\n            numerical stability (default: 1e-8)\n        weight_decay (float, optional): weight decay factor (default: 0)\n        amsgrad (boolean, optional): whether to use the AMSGrad variant of this\n            algorithm from the paper `On the Convergence of Adam and Beyond`_\n            (default: False)\n\n    .. _Adam\\: A Method for Stochastic Optimization:\n        https://arxiv.org/abs/1412.6980\n    .. _Fixing Weight Decay Regularization in Adam:\n        https://arxiv.org/abs/1711.05101\n    .. _On the Convergence of Adam and Beyond:\n        https://openreview.net/forum?id=ryQu7f-RZ\n    '

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        if (not (0.0 <= lr)):
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if (not (0.0 <= eps)):
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if (not (0.0 <= betas[0] < 1.0)):
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0]))
        if (not (0.0 <= betas[1] < 1.0)):
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        'Performs a single optimization step.\n\n        Arguments:\n            closure (callable, optional): A closure that reevaluates the model\n                and returns the loss.\n        '
        loss = None
        if (closure is not None):
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if (p.grad is None):
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                state = self.state[p]
                if (len(state) == 0):
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                (exp_avg, exp_avg_sq) = (state['exp_avg'], state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                (beta1, beta2) = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_((1 - beta1), grad)
                exp_avg_sq.mul_(beta2).addcmul_((1 - beta2), grad, grad)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = (1 - (beta1 ** state['step']))
                bias_correction2 = (1 - (beta2 ** state['step']))
                step_size = (
                    (group['lr'] * math.sqrt(bias_correction2)) / bias_correction1)
                if (group['weight_decay'] != 0):
                    p.data.add_((- group['weight_decay']), p.data)
                p.data.addcdiv_((- step_size), exp_avg, denom)
        return loss


class ExpAnnealLR(_LRScheduler):
    'Exponentially anneal the learning rate of each parameter group\n    from the initial lr to end_lr over a number of iterations.\n\n    Args:\n        optimizer (Optimizer): Wrapped optimizer.\n        end_lr (float): The final learning rate.\n        iterations (int): The number of iterations over which to increase the\n            learning rate.\n        last_epoch (int): The index of the last iteration. Default: -1.\n    '

    def __init__(self, optimizer, end_lr, iterations, last_epoch=(- 1)):
        self.end_lr = end_lr
        self.iterations = iterations
        super(ExpAnnealLR, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        iteration = (self.last_epoch + 1)
        pct = (iteration / self.iterations)
        return [(base_lr * ((self.end_lr / base_lr) ** pct)) for base_lr in self.base_lrs]


class ReduceLRWDOnPlateau(ReduceLROnPlateau):
    "Reduce learning rate and weight decay when a metric has stopped\n    improving. Models often benefit from reducing the learning rate by\n    a factor of 2-10 once learning stagnates. This scheduler reads a metric\n    quantity and if no improvement is seen for a 'patience' number\n    of epochs, the learning rate and weight decay factor is reduced for\n    optimizers that implement the the weight decay method from the paper\n    `Fixing Weight Decay Regularization in Adam`_.\n\n    .. _Fixing Weight Decay Regularization in Adam:\n        https://arxiv.org/abs/1711.05101\n\n    Args:\n        optimizer (Optimizer): Wrapped optimizer.\n        mode (str): One of `min`, `max`. In `min` mode, lr will\n            be reduced when the quantity monitored has stopped\n            decreasing; in `max` mode it will be reduced when the\n            quantity monitored has stopped increasing. Default: 'min'.\n        factor (float): Factor by which the learning rate will be\n            reduced. new_lr = lr * factor. Default: 0.1.\n        patience (int): Number of epochs with no improvement after\n            which learning rate will be reduced. For example, if\n            `patience = 2`, then we will ignore the first 2 epochs\n            with no improvement, and will only decrease the LR after the\n            3rd epoch if the loss still hasn't improved then.\n            Default: 10.\n        verbose (bool): If ``True``, prints a message to stdout for\n            each update. Default: ``False``.\n        threshold (float): Threshold for measuring the new optimum,\n            to only focus on significant changes. Default: 1e-4.\n        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,\n            dynamic_threshold = best * ( 1 + threshold ) in 'max'\n            mode or best * ( 1 - threshold ) in `min` mode.\n            In `abs` mode, dynamic_threshold = best + threshold in\n            `max` mode or best - threshold in `min` mode. Default: 'rel'.\n        cooldown (int): Number of epochs to wait before resuming\n            normal operation after lr has been reduced. Default: 0.\n        min_lr (float or list): A scalar or a list of scalars. A\n            lower bound on the learning rate of all param groups\n            or each group respectively. Default: 0.\n        eps (float): Minimal decay applied to lr. If the difference\n            between new and old lr is smaller than eps, the update is\n            ignored. Default: 1e-8.\n\n    Example:\n        >>> optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=1e-3)\n        >>> scheduler = ReduceLRWDOnPlateau(optimizer, 'min')\n        >>> for epoch in range(10):\n        >>>     train(...)\n        >>>     val_loss = validate(...)\n        >>>     # Note that step should be called after validate()\n        >>>     scheduler.step(val_loss)\n    "

    def step(self, metrics, epoch=None):
        current = metrics
        if (epoch is None):
            epoch = self.last_epoch = (self.last_epoch + 1)
        self.last_epoch = epoch
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if (self.num_bad_epochs > self.patience):
            self._reduce_lr(epoch)
            self._reduce_weight_decay(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_weight_decay(self, epoch):
        for (i, param_group) in enumerate(self.optimizer.param_groups):
            if (param_group['weight_decay'] != 0):
                old_weight_decay = float(param_group['weight_decay'])
                new_weight_decay = max(
                    (old_weight_decay * self.factor), self.min_lrs[i])
                if ((old_weight_decay - new_weight_decay) > self.eps):
                    param_group['weight_decay'] = new_weight_decay
                    if self.verbose:
                        log.info(''.join(['Epoch ', '{}'.format(epoch), ': reducing weight decay factor of group ', '{}'.format(
                            i), ' to ', '{:.4e}'.format(new_weight_decay), '.']))
