import torch
from tensordict import TensorClass
from active_adaptation.utils.math import clamp_norm
from typing import Tuple

class SpringForce(TensorClass):
    duration: torch.Tensor
    time: torch.Tensor # the time elapsed since the start of the force
    setpoint : torch.Tensor
    setpoint_mass: torch.Tensor
    setpoint_vel: torch.Tensor
    
    kp: torch.Tensor
    kd: torch.Tensor
    
    @classmethod
    def sample(cls, size: int, device: str):
        scalar = torch.empty(size, 1, device=device)
        offset = torch.zeros(size, 3, device=device)
        offset[:, 0] = -0.2
        return cls(
            duration=scalar.uniform_(3., 5.).clone(),
            time=torch.zeros(size, 1, device=device),
            setpoint=offset,
            setpoint_mass=400.*torch.ones(size, 1, device=device),
            setpoint_vel=torch.zeros(size, 3, device=device),
            kp=scalar.uniform_(60., 80.).clone(),
            kd=scalar.uniform_(10., 20.).clone(),
            batch_size=size,
        )
    
    def is_valid(self):
        return self.time < self.duration

    def get_force(self, pos: torch.Tensor, vel: torch.Tensor):
        """Return the world-frame force."""
        force = self.kp * (self.setpoint - pos) - self.kd * vel
        force = clamp_norm(force, 0., 200.) * self.is_valid()
        return force
    
    def step(self, pos: torch.Tensor, vel: torch.Tensor, dt: float):
        self.time.add_(dt)
        self.setpoint_vel.mul_(0.95).add_(-self.get_force(pos, vel) / self.setpoint_mass * dt)
        self.setpoint.add_(self.setpoint_vel * dt)


class ConstantForce(TensorClass):
    duration: torch.Tensor
    time: torch.Tensor # the time elapsed since the start of the force
    offset: torch.Tensor
    force: torch.Tensor
    
    @classmethod
    def sample(
        cls,
        size: int,
        force_scales: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        force_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        duration_range: Tuple[float, float] = (1.0, 4.0),
        device: str = "cpu",
    ):
        duration = torch.zeros(size, 1, device=device)
        duration.uniform_(*duration_range)
        offset = torch.rand(size, 3, device=device) * 2. - 1.
        offset *= torch.as_tensor(force_offsets, device=device)
        force = torch.rand(size, 3, device=device) * 2. - 1.
        force *= torch.as_tensor(force_scales, device=device)
        return cls(
            duration=duration,
            time=torch.zeros(size, 1, device=device),
            offset=offset,
            force=force,
            batch_size=size,
        )

    def get_force(self):
        """Return the world-frame force."""
        return self.force * (self.time < self.duration)


class ImpulseForce(TensorClass):
    duration: torch.Tensor
    time: torch.Tensor # the time elapsed since the start of the force
    peak: torch.Tensor

    @classmethod
    def sample(
        cls,
        size: int,
        device: str,
        impulse_scale: Tuple[float, float, float] = (100., 100., 20.),
        duration_range: Tuple[float, float] = (0.40, 0.60),
    ):
        with torch.device(device):
            duration = torch.empty(size, 1)
            duration.uniform_(*duration_range)
            impulse = torch.empty(size, 3)
            for i in range(3):
                impulse[:, i].uniform_(0, impulse_scale[i])
            impulse *= (torch.rand(size, 3) - 0.5).sign()

        # impule = peak * duration / 2
        peak = 2 * impulse / duration
        return cls(
            duration=duration,
            time=torch.zeros(size, 1, device=device),
            peak=peak,
            batch_size=size,
        )

    def get_force(self):
        """Return the world-frame force."""
        t = (self.time / self.duration).clamp(0., 1.)
        force = torch.where(t < 0.5, t * 2 * self.peak, (1 - t) * 2 * self.peak)
        return force