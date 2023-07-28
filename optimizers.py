import numpy as np

class Gauss_smooth():
    """Create the base line functions for Gaussian smoothing
    Args:
        fn_value: the function to calculate target pdf value, takes input (simulation_num, sample_size, d) and output (simulation_num, sample_size,)
        simulation_num: number of Monte Carlo simulations
        scale: the coefficient of noise
        x: the sample of data, have size (sample_size, d)
    """
    def __init__(self, fn_value, simulation_num, scale):
        self.simulation_num = simulation_num
        self.scale = scale
        self.fn_value = fn_value

    # get the reference function value, input size: (sample_size, d), output size (sample_size,)
    def get_ref(self, x):
        noise = np.random.normal(0,1,tuple([self.simulation_num]) + x.shape)
        ref_sum = self.fn_value(np.array([x]*self.simulation_num)+self.scale*noise)
        return np.mean(ref_sum, axis=0)

    # get the gradient of the reference function, input size: (sample_size, d), output size (sample_size, d)
    def get_ref_grad(self, x):
        noise = np.random.normal(0,1,tuple([self.simulation_num]) + x.shape)
        ref_grad_sum = noise * self.fn_value(np.array([x]*self.simulation_num)+self.scale*noise)[...,None]
        return np.mean(ref_grad_sum, axis=0,) * (2/self.scale)

class LD_Gauss_smooth(Gauss_smooth):
# Overdamped LD with Gaussian smoothing
    def __init__(self, fn_value, simulation_num=300, scale=1, step_size=0.1):
        super().__init__(fn_value, simulation_num, scale)
        self.step_size = step_size
        self.name = "LD"

    # one step of updating the vect
    def update_step(self, vect, grad):
        drift = -self.step_size * grad
        diffusion = np.random.normal(0,1,vect.shape) * (2*self.step_size)**0.5
        update = drift + diffusion
        return update

class anchored_LD_Gauss_smooth(Gauss_smooth):
# Anchored LD with Gaussian smoothing
    def __init__(self, fn_value, simulation_num=300, scale=1, step_size=0.1):
        super().__init__(fn_value, simulation_num, scale)
        self.step_size = step_size
        self.name = "Anchored LD"

    # one step of updating the vect
    def update_step(self, vect, grad):
        # get f and ref values, each has size (sample_size,)
        f_value = self.fn_value(vect[None,...])[0,:]
        ref_value = self.get_ref(vect)

        drift = -self.step_size * grad * np.exp(f_value-ref_value)[:,None]
        diffusion = np.random.normal(0,1,vect.shape) * np.exp((f_value-ref_value)/2)[:,None] * (2*self.step_size)**0.5
        update = drift + diffusion
        return update

class time_change_LD_Gauss_smooth(Gauss_smooth):
# Time change anchored LD with Gaussian smoothing
    def __init__(self, fn_value, simulation_num=300, scale=1, step_size=0.1):
        super().__init__(fn_value, simulation_num, scale)
        self.step_size = step_size
        self.name = "Time change anchored LD"

    # one step of updating the vect
    def update_step(self, vect, grad):
        # get f and ref values, each has size (sample_size,)
        f_value = self.fn_value(vect[None,...])[0,:]
        ref_value = self.get_ref(vect)

        step = np.exp(f_value-ref_value)[:,None] * self.step_size
        drift = -step * grad
        diffusion = np.random.normal(0,1,vect.shape) * (2*step)**0.5
        update = drift + diffusion
        return update
    
class Deterministic_smooth():
    """Create the base line functions for determinisitic smoothing, specifically for the regularizers
    Args:
        fn_value: the function to calculate target pdf value, takes input (1, sample_size, d) and output (1, sample_size,)
        scale: the coefficient of noise
        x: the sample of data, have size (sample_size, d)
    """
    def __init__(self, fn_value, ref_value, grad_value):
        self.fn_value = fn_value
        # get the reference function value, input size: (sample_size, d), output size (sample_size,)
        self.get_ref = ref_value
        # get the gradient of the reference function, input size: (sample_size, d), output size (sample_size, d)
        self.get_ref_grad = grad_value
    
class LD_reg_smooth(Deterministic_smooth):
# Overdamped LD with Gaussian smoothing
    def __init__(self, fn_value, ref_value, grad_value, step_size=0.1):
        super().__init__(fn_value, ref_value, grad_value)
        self.step_size = step_size
        self.name = "LD"

    # one step of updating the vect
    def update_step(self, vect, grad):
        drift = -self.step_size * grad
        diffusion = np.random.normal(0,1,vect.shape) * (2*self.step_size)**0.5
        update = drift + diffusion
        return update

class anchored_LD_reg_smooth(Deterministic_smooth):
# Anchored LD with Gaussian smoothing
    def __init__(self, fn_value, ref_value, grad_value, step_size=0.1):
        super().__init__(fn_value, ref_value, grad_value)
        self.step_size = step_size
        self.name = "Anchored LD"

    # one step of updating the vect
    def update_step(self, vect, grad):
        # get f and ref values, each has size (sample_size,)
        f_value = self.fn_value(vect[None,...])[0,:]
        ref_value = self.get_ref(vect)

        drift = -self.step_size * grad * np.exp(f_value-ref_value)[:,None]
        diffusion = np.random.normal(0,1,vect.shape) * np.exp((f_value-ref_value)/2)[:,None] * (2*self.step_size)**0.5
        update = drift + diffusion
        return update

class time_change_LD_reg_smooth(Deterministic_smooth):
# Time change anchored LD with Gaussian smoothing
    def __init__(self, fn_value, ref_value, grad_value, step_size=0.1):
        super().__init__(fn_value, ref_value, grad_value)
        self.step_size = step_size
        self.name = "Time change anchored LD"

    # one step of updating the vect
    def update_step(self, vect, grad):
        # get f and ref values, each has size (sample_size,)
        f_value = self.fn_value(vect[None,...])[0,:]
        ref_value = self.get_ref(vect)

        step = np.exp(f_value-ref_value)[:,None] * self.step_size
        drift = -step * grad
        diffusion = np.random.normal(0,1,vect.shape) * (2*step)**0.5
        update = drift + diffusion
        return update
    
class vanilla_LD_Gauss_smooth():
# Overdamped LD with Gaussian smoothing
    def __init__(self, step_size=0.1):
        self.step_size = step_size
        self.name = "LD"

    # one step of updating the vect
    # vect has shape (sample_size, ., .)
    def update_step(self, vect, grad, loss, ref):
        drift = -self.step_size * grad
        diffusion = np.random.normal(0,1,vect.shape) * (2*self.step_size)**0.5
        update = drift + diffusion
        return update

class vanilla_anchored_LD_Gauss_smooth():
# Anchored LD with Gaussian smoothing
    def __init__(self, step_size=0.1):
        self.step_size = step_size
        self.name = "Anchored LD"

    # one step of updating the vect
    def update_step(self, vect, grad, loss, ref):
        # get f and ref values, each has size (sample_size,)
        # vect has shape (sample_size, ., .)
        drift = -self.step_size * grad * np.exp(loss-ref)[:,None,None]
        diffusion = np.random.normal(0,1,vect.shape) * np.exp((loss-ref)/2)[:,None,None] * (2*self.step_size)**0.5
        update = drift + diffusion
        return update

class vanilla_time_change_LD_Gauss_smooth():
# Time change anchored LD with Gaussian smoothing
    def __init__(self, step_size=0.1):
        self.step_size = step_size
        self.name = "Time change anchored LD"

    # one step of updating the vect
    def update_step(self, vect, grad, loss, ref):
        # get f and ref values, each has size (sample_size,)
        # vect has shape (sample_size, ., .)
        step = np.exp(loss-ref)[:,None,None] * self.step_size
        drift = -step * grad
        diffusion = np.random.normal(0,1,vect.shape) * (2*step)**0.5
        update = drift + diffusion
        return update