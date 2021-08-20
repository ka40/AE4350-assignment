import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.arguments import get_args

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
#changes
class IAMPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, env, base=None, base_kwargs=None):
        super(IAMPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                if base_kwargs['IAM']:
                    hxs_size = 113
                    base = atariBase # Breakout
                else:
                    hxs_size = 113
                    base = CNNBase
            elif len(obs_shape) == 1:
                if env == 'warehouse':
                    base = warehouseBase
                    if base_kwargs['IAM']:
                        hxs_size = 25
                    else:
                        hxs_size = 73
    #end changes
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], hxs_size, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

#from standard
class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs



#changes
class IAMBase(nn.Module):
    def __init__(self, recurrent, IAM, recurrent_input_size, hidden_size):
        super(IAMBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._IAM = IAM
        #end changes

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent
    
    #changes
    @property
    def is_IAM(self):
        return self._IAM
    #end changes

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

#changes
class warehouseBase(IAMBase):
    """
    For Warehouse environment, the IAM architecture looks like this:
    In essence, the fnn receives the whole observation, 
    and the gru RNN receives only the necessary information

    obs >   [fnn ]             >|> |nn  | >value
            |____|              |  |____|
        >  [dset]  -> |gru | ->|-> |nn  | ->dist()->mode()/sample()->action 

    The inputs and outputs of each piece in the architecture is as follows;
    observation, obs: (num_processes, num_inputs: 73 in warehouse)
    fnn output: (num_processes, hidden_size_fnn)
    dset output: (num_processes, 25)
    gru output: ((num_processes, hidden_size_gru), rnn_hxs)
    output_size:  hidden_size_fnn plus hidden_size_gru
    """
    def __init__(self, num_inputs, hxs_size, recurrent=False, IAM=False, hidden_size=64):
        super(warehouseBase, self).__init__(recurrent, IAM, hxs_size, hidden_size)

        # dset attributes of original model
        self.dset = [0, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
           63, 64, 65, 66, 67, 68, 69, 70, 71, 72]

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(2*hidden_size, hidden_size)))

        self.critic = nn.Sequential(
            init_(nn.Linear(2*hidden_size, 1)))

        # critic for non IAM architecture
        self.critic_normal = nn.Sequential(
            init_(nn.Linear(hidden_size, 1)))

        self.fnn = nn.Sequential(
            init_(nn.Linear(num_inputs, 512)),nn.ReLU(),
            init_(nn.Linear(512, 256)),nn.ReLU(),
            init_(nn.Linear(256, hidden_size)),nn.ReLU())

        self.train()

    def manual_dpatch(self, obs_input):
        """
        inf_hidden is the input of recurrent net which is a subset of the observation
        dset is manually set and is static(does not change from frame to frame).
        """
        
        inf_hidden = obs_input[:, self.dset]

        return inf_hidden

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        # go through d_pathched reccurent network, update for one step
        if self.is_recurrent:
            if self.is_IAM:
                x_rec = self.manual_dpatch(x)
                x_rec, rnn_hxs = self._forward_gru(x_rec, rnn_hxs, masks)
                fnn_out = self.fnn(x)
                x = torch.cat((x_rec,fnn_out), 1)
            else:
                x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        else:
            x = self.fnn(x)
        
        if self.is_IAM:
            hidden_critic = self.critic(x)
            hidden_actor = self.actor(x)
        else:
            hidden_critic = self.critic_normal(x)
            hidden_actor = x

        return hidden_critic, hidden_actor, rnn_hxs

class atariBase(IAMBase):
    """
    For the atari games, there is an image observed environment. 
    This is also the case for breakout.
    The IAM architecture changes to the following:
    The FNN receives a flattened image that went through a cnn.
    The GRU receives only the filtered dset

    obs > [cnn ] >  [> flatten() >  [fnn ]   >|> [nn  ] >value
          [____]    [               [____]    |  [____]
                    [   [atte]                |
                    [>  [tion]   >  [gru ]   >|> [nn  ] >dist()->mode()/sample()->action 
                        [____]      [____]       [____]   
    """
    def __init__(self, num_inputs, hxs_size, recurrent=False, IAM=False, hidden_size=64):
        super(atariBase, self).__init__(recurrent, IAM, hxs_size, hidden_size)
        self._depatch_size = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU())

        self.fnn = nn.Sequential(
            Flatten(),
            init_(nn.Linear(64 * 7 * 7, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.ReLU())
        

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.actor = nn.Sequential(
            init_(nn.Linear(2*hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(2*hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        # functional layers
        self.dpatch_conv = init_(nn.Linear(64, 128)) #depatch, merge the channels and encode them

        self.dpatch_auto = init_(nn.Linear(64, 128))
        self.dpatch_auto_norm = init_(nn.Linear(7*7*128, 128))

        self.dpatch_prehidden = init_(nn.Linear(hidden_size, 128))

        self.dpatch_combine = nn.Tanh()

        self.dpatch_weights = nn.Sequential(
            init_(nn.Linear(128,1)), nn.Softmax(dim=1))

        self.train()

    def attention(self, hidden_conv, rnn_hxs):
        hidden_conv = hidden_conv.permute(0,2,3,1)
        shape = hidden_conv.size()
        num_regions = shape[1]*shape[2]
        hidden = torch.reshape(hidden_conv, ([-1,num_regions,shape[3]]))
        linear_conv = self.dpatch_conv(hidden)        
        linear_prehidden = self.dpatch_prehidden(rnn_hxs)
        context = self.dpatch_combine(linear_conv + torch.unsqueeze(linear_prehidden, 1))
        attention_weights = self.dpatch_weights(context)
        dpatch = torch.sum(attention_weights*hidden,dim=1)
        inf_hidden = torch.cat((dpatch,torch.reshape(attention_weights, ([-1, num_regions]))), 1)

        return inf_hidden   

    def forward(self, inputs, rnn_hxs, masks):
        hidden_conv = self.cnn(inputs / 255.0)

        fnn_out = self.fnn(hidden_conv)
        inf_hidden = self.attention(hidden_conv, rnn_hxs)
        rnn_out, rnn_hxs = self._forward_gru(inf_hidden, rnn_hxs, masks)

        x = torch.cat((rnn_out,fnn_out), 1)

        hidden_critic = self.critic(x) 
        hidden_actor = self.actor(x)

        return hidden_critic, hidden_actor, rnn_hxs
#end changes