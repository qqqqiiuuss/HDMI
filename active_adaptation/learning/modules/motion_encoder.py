"""
TWIST-style Motion Encoder for temporal sequence processing

This module implements the 1D CNN-based motion encoder from TWIST paper,
adapted for HDMI's observation format (past + current + future frames).

Key differences from TWIST:
- HDMI: 21 frames (past 10 + current 1 + future 10)
- TWIST: 20 frames (future only)

The encoder compresses high-dimensional temporal sequences into compact latent representations.
"""

import torch
import torch.nn as nn


class MotionEncoder1D(nn.Module):
    """
    1D Convolutional Motion Encoder

    Encodes temporal motion sequences using 1D CNN to extract motion patterns.
    Automatically adapts network architecture based on the number of timesteps.

    Args:
        activation_fn: Activation function (default: nn.ELU)
        input_size: Dimension of each timestep's observation
        tsteps: Number of timesteps in the sequence
        output_size: Dimension of output latent vector
        tanh_encoder_output: Whether to apply tanh to output (default: False)

    Forward:
        Input: [batch, tsteps * input_size] - Flattened temporal sequence
        Output: [batch, output_size] - Compressed latent representation

    Example:
        >>> encoder = MotionEncoder1D(nn.ELU(), input_size=32, tsteps=21, output_size=128)
        >>> obs = torch.randn(4096, 21 * 32)  # [batch, 21 timesteps × 32 dims]
        >>> latent = encoder(obs)  # [4096, 128]
    """

    def __init__(
        self,
        activation_fn=None,
        input_size=32,
        tsteps=21,
        output_size=128,
        tanh_encoder_output=False
    ):
        super().__init__()

        if activation_fn is None:
            activation_fn = nn.ELU()

        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.input_size = input_size
        self.output_size = output_size

        channel_size = 20  # Base channel size (TWIST default)

        # Step 1: Per-frame linear projection
        # Projects each timestep's observation from input_size → 3*channel_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size),
            self.activation_fn,
        )

        # Step 2: 1D Temporal Convolution layers
        # Architecture adapts based on tsteps
        if tsteps == 50:
            # For long sequences (50 timesteps)
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3*channel_size, out_channels=2*channel_size,
                         kernel_size=8, stride=4),
                self.activation_fn,
                nn.Conv1d(in_channels=2*channel_size, out_channels=channel_size,
                         kernel_size=5, stride=1),
                self.activation_fn,
                nn.Conv1d(in_channels=channel_size, out_channels=channel_size,
                         kernel_size=5, stride=1),
                self.activation_fn,
                nn.Flatten()
            )
        elif tsteps == 21 or tsteps == 20:
            # For medium sequences (20-21 timesteps) - HDMI default
            # Receptive field covers ~16 timesteps
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3*channel_size, out_channels=2*channel_size,
                         kernel_size=6, stride=2),  # Output: (21-6)/2+1 = 8 timesteps
                self.activation_fn,
                nn.Conv1d(in_channels=2*channel_size, out_channels=channel_size,
                         kernel_size=4, stride=2),  # Output: (8-4)/2+1 = 3 timesteps
                self.activation_fn,
                nn.Flatten()
            )
        elif tsteps == 10:
            # For short sequences (10 timesteps)
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3*channel_size, out_channels=2*channel_size,
                         kernel_size=4, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2*channel_size, out_channels=channel_size,
                         kernel_size=2, stride=1),
                self.activation_fn,
                nn.Flatten()
            )
        elif tsteps == 1:
            # Degenerate case: Single timestep (no temporal convolution needed)
            self.conv_layers = nn.Flatten()
        else:
            raise ValueError(f"tsteps must be 1, 10, 20, 21, or 50. Got {tsteps}")

        # Step 3: Output projection
        # Maps flattened conv features to output_size
        self.linear_output = nn.Linear(channel_size * 3, output_size)

        self.tanh_encoder_output = tanh_encoder_output
        if tanh_encoder_output:
            self.tanh = nn.Tanh()

    def forward(self, obs):
        """
        Forward pass through motion encoder

        Args:
            obs: [batch_size, tsteps * input_size] - Flattened temporal observation

        Returns:
            output: [batch_size, output_size] - Compressed motion latent

        Processing steps:
        1. Reshape: [B, T*D] → [B*T, D]
        2. Per-frame projection: [B*T, D] → [B*T, 60]
        3. Reshape for Conv1D: [B*T, 60] → [B, 60, T]
        4. 1D Convolution: [B, 60, T] → [B, 20, 3] → [B, 60]
        5. Output projection: [B, 60] → [B, output_size]
        """
        batch_size = obs.shape[0]
        T = self.tsteps

        # DEBUG: Print input info
        if not hasattr(self, '_forward_debug_printed'):
            print(f"\n[MotionEncoder.forward DEBUG]:")
            print(f"  obs.shape = {obs.shape}")
            print(f"  obs.numel() = {obs.numel()}")
            print(f"  batch_size = {batch_size}")
            print(f"  T (tsteps) = {T}")
            print(f"  input_size = {self.input_size}")
            print(f"  Expected obs.shape = [{batch_size}, {T * self.input_size}]")
            print(f"  About to reshape to: [{batch_size * T}, -1]")
            print(f"  That would be: [{batch_size * T}, {obs.numel() // (batch_size * T)}]")
            self._forward_debug_printed = True

        # Step 1: Per-frame projection
        # Reshape to [batch * tsteps, input_size] for parallel processing
        projection = self.encoder(obs.reshape([batch_size * T, -1]))
        # Output: [batch * tsteps, 3*channel_size]

        # Step 2: Reshape for 1D convolution
        # Conv1d expects [batch, channels, time]
        projection = projection.reshape([batch_size, T, -1]).permute((0, 2, 1))
        # Output: [batch, 3*channel_size, tsteps]

        # Step 3: Temporal convolution
        output = self.conv_layers(projection)
        # Output: [batch, channel_size * 3] (after flatten)

        # Step 4: Output projection
        output = self.linear_output(output)
        # Output: [batch, output_size]

        # Optional: Apply tanh activation
        if self.tanh_encoder_output:
            output = self.tanh(output)

        return output


class MotionEncoderRNN(nn.Module):
    """
    RNN-based Motion Encoder (Alternative to 1D CNN)

    Uses LSTM to encode temporal sequences. Can be used as a drop-in replacement
    for MotionEncoder1D for comparison experiments.

    Args:
        input_size: Dimension of each timestep's observation
        hidden_size: Hidden state dimension of LSTM
        output_size: Dimension of output latent vector
        num_layers: Number of LSTM layers (default: 2)
    """

    def __init__(self, input_size=32, hidden_size=128, output_size=128, num_layers=2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, obs):
        """
        Forward pass through RNN encoder

        Args:
            obs: [batch_size, tsteps * input_size] - Flattened temporal observation

        Returns:
            output: [batch_size, output_size] - Encoded latent
        """
        batch_size = obs.shape[0]
        tsteps = obs.shape[1] // self.input_size

        # Reshape to [batch, tsteps, input_size]
        obs_reshaped = obs.reshape(batch_size, tsteps, self.input_size)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(obs_reshaped)

        # Use final hidden state
        final_hidden = h_n[-1]  # [batch, hidden_size]

        # Project to output size
        output = self.output_proj(final_hidden)

        return output


def test_motion_encoder():
    """Test function to verify MotionEncoder correctness"""
    print("Testing MotionEncoder1D...")

    # Test with HDMI config (21 timesteps)
    batch_size = 4096
    input_size = 32  # Single frame dim (root_pos + root_ori + joint_pos)
    tsteps = 21      # past 10 + current 1 + future 10
    output_size = 128

    encoder = MotionEncoder1D(
        activation_fn=nn.ELU(),
        input_size=input_size,
        tsteps=tsteps,
        output_size=output_size
    )

    # Create dummy input
    obs = torch.randn(batch_size, tsteps * input_size)

    # Forward pass
    latent = encoder(obs)

    print(f"Input shape: {obs.shape}")
    print(f"Output shape: {latent.shape}")
    print(f"Expected output shape: [{batch_size}, {output_size}]")

    assert latent.shape == (batch_size, output_size), "Output shape mismatch!"
    print("✓ MotionEncoder1D test passed!")

    # Test parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test with TWIST config (20 timesteps)
    encoder_twist = MotionEncoder1D(
        activation_fn=nn.ELU(),
        input_size=58,  # TWIST single frame dim
        tsteps=20,
        output_size=128
    )
    obs_twist = torch.randn(4096, 20 * 58)
    latent_twist = encoder_twist(obs_twist)
    print(f"\nTWIST config - Input: {obs_twist.shape}, Output: {latent_twist.shape}")
    print("✓ TWIST compatibility test passed!")


if __name__ == "__main__":
    test_motion_encoder()
