# from https://github.com/chengxuxin/extreme-parkour

import torch


def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)


def calculate_conv_output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def calculate_pool_output_size(input_size, kernel_size, stride):
    return (input_size - kernel_size) // stride + 1


def compute_latent_dim_size(input_height, input_width, conv1_kernel=5, conv1_stride=1, conv1_padding=0, pool_kernel=2,
                            pool_stride=2, conv2_kernel=3, conv2_stride=1, conv2_padding=0):
    # First Convolution
    height_after_conv1 = calculate_conv_output_size(input_height, conv1_kernel, conv1_stride, conv1_padding)
    width_after_conv1 = calculate_conv_output_size(input_width, conv1_kernel, conv1_stride, conv1_padding)

    # Max Pooling
    height_after_pool = calculate_pool_output_size(height_after_conv1, pool_kernel, pool_stride)
    width_after_pool = calculate_pool_output_size(width_after_conv1, pool_kernel, pool_stride)

    # Second Convolution
    height_after_conv2 = calculate_conv_output_size(height_after_pool, conv2_kernel, conv2_stride, conv2_padding)
    width_after_conv2 = calculate_conv_output_size(width_after_pool, conv2_kernel, conv2_stride, conv2_padding)

    # Flattened size
    return 64 * height_after_conv2 * width_after_conv2
