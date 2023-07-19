import numpy as np
import torch
from torch import nn

from dsslac.network.initializer import initialize_weight
from dsslac.utils import build_mlp, reparameterize


class LatentGaussianPolicy(torch.jit.ScriptModule):
    """
    Policy parameterized as diagonal gaussian distribution.
    """

    def __init__(self, action_shape, z1_dim, z2_dim, hidden_units=(256, 256)):
        super(LatentGaussianPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        self.net = build_mlp(
            input_dim=z1_dim+z2_dim,
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(inplace=True, negative_slope=0.2),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, feature_action):
        means = torch.chunk(self.net(feature_action), 2, dim=-1)[0]
        return torch.tanh(means)

    @torch.jit.script_method
    def sample(self, feature_action):
        mean, log_std = torch.chunk(self.net(feature_action), 2, dim=-1)
        action, log_pi = reparameterize(mean, log_std.clamp(-20, 2))
        return action, log_pi



class GaussianPolicy(torch.jit.ScriptModule):
    """
    Policy parameterized as diagonal gaussian distribution.
    """

    def __init__(self, action_shape, num_sequences, feature_dim, hidden_units=(256, 256)):
        super(GaussianPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        self.net = build_mlp(
            input_dim=num_sequences * feature_dim + (num_sequences - 1) * action_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=(512,)+hidden_units,
            hidden_activation=nn.LeakyReLU(inplace=True, negative_slope=0.2),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, feature_action):
        means = torch.chunk(self.net(feature_action), 2, dim=-1)[0]
        return torch.tanh(means)

    @torch.jit.script_method
    def sample(self, feature_action):
        mean, log_std = torch.chunk(self.net(feature_action), 2, dim=-1)
        action, log_pi = reparameterize(mean, log_std.clamp(-20, 2))
        return action, log_pi


class TwinnedQNetwork(torch.jit.ScriptModule):
    """
    Twinned Q networks.
    """

    def __init__(
        self,
        action_shape,
        z1_dim,
        z2_dim,
        hidden_units=(256, 256),
        disable_twin=False
    ):
        super(TwinnedQNetwork, self).__init__()

        self.net1 = build_mlp(
            input_dim=action_shape[0] + z1_dim + z2_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(inplace=True, negative_slope=0.2),
        ).apply(initialize_weight)
        if disable_twin:
            self.net2 = self.net1
        else:
            self.net2 = build_mlp(
                input_dim=action_shape[0] + z1_dim + z2_dim,
                output_dim=1,
                hidden_units=hidden_units,
                hidden_activation=nn.LeakyReLU(inplace=True, negative_slope=0.2),
            ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, z, action):
        x = torch.cat([z, action], dim=1)
        return self.net1(x), self.net2(x)

class SingleQNetwork(torch.jit.ScriptModule):
    """
    Twinned Q networks.
    """

    def __init__(
        self,
        action_shape,
        z1_dim,
        z2_dim,
        hidden_units=(256, 256),
        init_output=0
    ):
        super(SingleQNetwork, self).__init__()

        self.net1 = build_mlp(
            input_dim=action_shape[0] + z1_dim + z2_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(inplace=True, negative_slope=0.2),
            output_activation = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        ).apply(initialize_weight)

        torch.nn.init.constant_(self.net1._modules["last_linear"]._parameters["bias"], init_output)
      

    @torch.jit.script_method
    def forward(self, z, action):
        x = torch.cat([z, action], dim=1)
        return self.net1(x)


class CosineEmbeddingNetwork(nn.Module):

    def __init__(
        self,
        action_shape,
        z1_dim,
        z2_dim,
        num_cosines=64
    ):
        super(CosineEmbeddingNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_cosines, action_shape[0] + z1_dim + z2_dim),
            nn.ReLU()
        ).apply(initialize_weight)
        self.num_cosines = num_cosines
        self.embedding_dim = action_shape[0] + z1_dim + z2_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
            ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)

        return tau_embeddings


class QuantileNetwork(nn.Module):

    def __init__(
        self,
        action_shape,
        z1_dim,
        z2_dim,
        init_output=0
    ):
        super(QuantileNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(action_shape[0] + z1_dim + z2_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            ).apply(initialize_weight)

        self.num_actions = action_shape[0] 
        self.embedding_dim = action_shape[0] + z1_dim + z2_dim

    def forward(self, z, action, tau_embeddings):
        state_embeddings = torch.cat([z, action], dim=1)
        assert state_embeddings.shape[0] == tau_embeddings.shape[0]
        assert state_embeddings.shape[1] == tau_embeddings.shape[2]

        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = state_embeddings.view(
            batch_size, 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size * N, self.embedding_dim)

        # Calculate quantile values.
        quantiles = self.net(embeddings)

        return quantiles.view(batch_size, N, 1)
