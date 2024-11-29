# https://github.com/chengxuxin/extreme-parkour

import torch
import torch.nn as nn

from ..utils import compute_latent_dim_size


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, n_proprio=None, ignore_yaw=False) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if n_proprio == None:
            self.combination_mlp = nn.Sequential(
                nn.Linear(32 + 53, 128),
                activation,
                nn.Linear(128, 32)
            )
        else:
            self.combination_mlp = nn.Sequential(
                nn.Linear(32 + n_proprio, 128),
                activation,
                nn.Linear(128, 32)
            )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)

        yaw_feat = 0 if ignore_yaw else 2

        self.output_mlp = nn.Sequential(
            nn.Linear(512, 32 + yaw_feat),
            last_activation
        )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))

        return depth_latent

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach().clone()


class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone, depth_buffer_len, n_proprio) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128),
            activation,
            nn.Linear(128, 32)
        )

        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=depth_buffer_len, out_channels=16, kernel_size=4, stride=2),
                                    # (30 - 4) / 2 + 1 = 14,
                                    activation,
                                    nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2),  # 14-2+1 = 13,
                                    activation)
        self.mlp = nn.Sequential(nn.Linear(16 * 14, 32),
                                 activation)

    def forward(self, depth_image, proprioception):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(None, depth_image.flatten(0, 1), None)  # [batch_size * num, 32]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 32]
        depth_latent = self.conv1d(depth_latent)
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent


class DepthOnlyFCBackbone(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, image_size, output_activation=None,
                 num_frames=1):
        super().__init__()

        assert len(image_size) == 2, "Please do not include the batch size or channel dimension in the image size."
        latent_dim_size = compute_latent_dim_size(*image_size)

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(latent_dim_size, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent
