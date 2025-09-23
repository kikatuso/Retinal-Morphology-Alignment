import numpy as np


class LambdaWarmUpCosineScheduler2():
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """
    def __init__(self,warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                    1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f
        
if __name__ =='__main__':

    from torch.optim.lr_scheduler import LambdaLR
    import torch.optim as optim
    from torch import randn

    scheduler =  LambdaLinearScheduler(
        warm_up_steps=[15000],
        f_min=[0.10],
        f_max=[1.],
        f_start=[0.01],
        cycle_lengths=[740400]
        
    )

    opt = optim.Adam([randn(2, 2, requires_grad=True)], lr=1e-3)
    scheduler2 = LambdaLR(opt, lr_lambda=scheduler.schedule)

    epochs = 300
    dataset_size = 78983
    batch_size = 32
    steps_per_epoch = dataset_size // batch_size
    n_steps = epochs * steps_per_epoch
    print(f"n_steps: {n_steps} for {epochs} epochs, dataset size {dataset_size}, batch size {batch_size}")

    lr_h = []
    for i in range(n_steps):
        
        #lr = scheduler(i)
        #lr = scheduler2(i)
        scheduler2.step()
        lr = opt.param_groups[0]['lr']

        lr_h.append(lr)
    
    import matplotlib.pyplot as plt

    plt.plot(lr_h)
    plt.xlabel("Training Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid()
    # set xticks at epoch boundaries
    epoch_ticks = [i * steps_per_epoch for i in range(0, epochs+1, max(1, epochs//10))]
    plt.xticks(epoch_ticks, [str(i) for i in range(0, epochs+1, max(1, epochs//10))])
    plt.savefig("lr_schedule.png")


    start = 10_000
    end   = start + steps_per_epoch 
    plt.figure()
    plt.plot(lr_h[start:end])
    plt.xlabel("Step"); plt.ylabel("LR"); plt.grid(True)
    plt.title(f"LR schedule zoomed in steps {start} to {end}")
    plt.savefig("lr_schedule_zoom.png")
