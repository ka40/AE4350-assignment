# This repo cloned pytorch-a2c-ppo-acktr

This repo is the codebase for the assignment of Kevin Bislip for AE4350.
In this assignment, the Influence-Aware Memory(IAM) architecture(https://arxiv.org/abs/1911.07643) was reproduced in Pytorch.
The original paper's source can be found in https://github.com/INFLUENCEorg/influence-aware-memory.
The repo is based from pytorch-a2c-ppo-acktr for its ppo framework and structure.

## Changes done to original repo:

1. Changing the hyperparameters in a2c_ppo_acktr\arguments.py to the ones used in [11].
2. Installing and registering the warehouse environment.
3. Implementing the models; IAM, changing the network sizes and layers of the FNN and the GRU to
match the paper’s.
4. Run all the combinations of environments and algorithms given in Table 2.1.
5. Make the visualization code ‘visualization.py’ and use it to visualize and compare the results with the
original paper.

## Requirements from the original repo

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [Stable baselines3](https://github.com/DLR-RM/stable-baselines3)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Other requirements
pip install -r requirements.txt
```

## Code to run models


### Warehouse
All ware
```bash
python main.py --env-name warehouse --num-steps 8 --recurrent-policy --log-dir ./log_w/FNN8-1 ; python main.py --env-name warehouse --num-steps 8 --recurrent-policy --log-dir ./log_w/FNN8-2 ; python main.py --env-name warehouse --num-steps 8 --recurrent-policy --log-dir ./log_w/FNN8-3 ; python main.py --env-name warehouse --num-steps 1 --recurrent-policy --log-dir ./log_w/FNN1-1 ; python main.py --env-name warehouse --num-steps 1 --recurrent-policy --log-dir ./log_w/FNN1-2 ; python main.py --env-name warehouse --num-steps 1 --recurrent-policy --log-dir ./log_w/FNN1-3 ; python main.py --env-name warehouse --num-steps 8 --log-dir ./log_w/GRU-1 ; python main.py --env-name warehouse --num-steps 8 --log-dir ./log_w/GRU-2 ; python main.py --env-name warehouse --num-steps 8 --log-dir ./log_w/GRU-3 ; python main.py --env-name warehouse --num-steps 8 --IAM --log-dir ./log_w/IAM-1 ; python main.py --env-name warehouse --num-steps 8 --IAM --log-dir ./log_w/IAM-2 ; python main.py --env-name warehouse --num-steps 8 --IAM --log-dir ./log_w/IAM-3
```

Run FNN8
```bash
python main.py --env-name warehouse --num-steps 8 --recurrent-policy --log-dir ./log_w/FNN8-1 --seed 1 12306 seconds
python main.py --env-name warehouse --num-steps 8 --recurrent-policy --log-dir ./log_w/FNN8-2 --seed 2
python main.py --env-name warehouse --num-steps 8 --recurrent-policy --log-dir ./log_w/FNN8-3 --seed 3 14635.87 seconds
```
Run FNN1
```bash
python main.py --env-name warehouse --num-steps 1 --recurrent-policy --log-dir ./log_w/FNN1-1 --seed 1 takes too long 170fps
```
Run GRU only
```bash
python main.py --env-name warehouse --num-steps 8 --log-dir ./log_w/GRU-1 --seed 1
python main.py --env-name warehouse --num-steps 8 --log-dir ./log_w/GRU-2 --seed 2
python main.py --env-name warehouse --num-steps 8 --log-dir ./log_w/GRU-3 --seed 3 12880 seconds
```
Run IAM
```bash
python main.py --env-name warehouse --num-steps 8 --IAM --log-dir ./log_w/IAM-1 --seed 1
python main.py --env-name warehouse --num-steps 8 --IAM --log-dir ./log_w/IAM-2 --seed 2
python main.py --env-name warehouse --num-steps 8 --IAM --log-dir ./log_w/IAM-3 --seed 3
```

### breakout no flicker 50e6
#### IAM
```bash
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 50007104 --num-steps 8 --lr 0.00025 --log-dir ./log_b/IAM-1 --IAM
'''

```bash
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 160000000 --num-steps 8 --lr 0.00025 --log-dir ./log_breakout_no_flicker/run2 --IAM
'''
log:
log_breakout_no_flicker/run2
11 hours 50e6
Updates 781350, num timesteps 50006464, FPS 1288 
 Last 10 training episodes: mean/median reward 342.1/378.5, min/max reward 138.0/423.0

Updates 781360, num timesteps 50007104, FPS 1288 
 Last 10 training episodes: mean/median reward 342.1/378.5, min/max reward 138.0/423.0
___________


### Breakout flicker

Run IAM with correct hyperparameters
```bash
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 50007104 --num-steps 8 --ppo-epoch 3 --lr 0.00025 --log-dir ./log_bf/IAM-1 --IAM --flicker 42667 seconds
```

### tuning learning rate breakout flicker
{10-2, 10−3, 10-4, 10-5}
```bash
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 10000000 --num-steps 8 --ppo-epoch 3 --lr 0.01 --log-dir ./log_bf_lr/IAM_lr_001-1 --IAM --flicker 9177 seconds
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0003 --log-dir ./log_bf_lr/IAM_lr_00003-1 --IAM --flicker
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0002 --log-dir ./log_bf_lr/IAM_lr_00002-1 --IAM --flicker
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0001 --log-dir ./log_bf_lr/IAM_lr_00001-1 --IAM --flicker 17857 seconds

python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0003 --log-dir ./log_bf_lr/IAM_lr_00003-1 --IAM --flicker ; python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0002 --log-dir ./log_bf_lr/IAM_lr_00002-1 --IAM --flicker ; python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0001 --log-dir ./log_bf_lr/IAM_lr_00001-1 --IAM --flicker
```

## Code implementation

Snippets of the code will be pasted for illustration and explanation, in order to aid reproduction. First, a base class inherited from nn.module was defined: 

class IAMBase(nn.Module):
    def __init__(self, recurrent, IAM, recurrent_input_size, hidden_size):
        super(IAMBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._IAM = IAM


The warehouseBase inherits from the IAMBase class and when creating an object the variables recurrent and IAM can be used to switch between architectures of IAM, plain GRU and FNN.


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


The static d-set is then implemented in the following snippet, by taking the column vectors of the observation matrix that correspond to the numbers found in the static dset list.


def manual_dpatch(self, obs_input):
    """
    inf_hidden is the input of recurrent net which is a subset of the observation
    dset is manually set and is static(does not change from frame to frame).
    """
    
    inf_hidden = obs_input[:, self.dset]

    return inf_hidden


Now that the modules and functions of the architecture are prepared, the flow of the architecture, or the forward function can be constructed:


def forward(self, inputs, rnn_hxs, masks):
    x = inputs

    # go through d_patched recurrent network, update for one step
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


Recalling that in the case that the algorithm chosen is IAM, self.\textunderscore recurrent and self.is\textunderscore IAM are both True. Hence, as illustrated in the diagram, the d-set extraction is fed into the RNN, whereas the FNN receives the whole observation matrix. Afterwards, they are concatenated and the value and policy are computed.

### Atari Breakout architecture

In this more complex environment, the input features the algorithm should focus on change over time. Hence, an attention mechanism is used instead of a manually implemented d-set. \\
A new class is defined for this, which also inherits from the class IAMBase. Additionally, a convolutional neural network is defined for image processing and a fully connected neural network is designed for the IAM. The entire structure of the neural network graph used in this case is also shown in a comment.


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

        # functional layers: depatch, merge the channels and encode them
        self.dpatch_conv = init_(nn.Linear(64, 128)) 
        self.dpatch_auto = init_(nn.Linear(64, 128))
        self.dpatch_auto_norm = init_(nn.Linear(7*7*128, 128))

        self.dpatch_prehidden = init_(nn.Linear(hidden_size, 128))
        self.dpatch_combine = nn.Tanh()

        self.dpatch_weights = nn.Sequential(
            init_(nn.Linear(128,1)), nn.Softmax(dim=1))

        self.train()


The method for the attention mechanism is the most important part. This method takes two inputs: hidden\textunderscore conv, the output from the CNN, and the hidden state of the recurrent neural network from the last time step, rnn\textunderscore hxs. The new observation matrix will first be reshaped into [batch size, width, height, channels] and the dimensions of width and height will be merged and it conserves the topology of the input. Afterwards, the weight matrix of all regions is computed from the combination of current input and past hidden state. Lastly, the weight matrix is decides which regions of the input observations should be passed into the recurrent neural network at current time step. In essence, the algorithm will learn where to look at according to the memory of the past states.


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

Here, the forward method is shown, in this case it was implemented only for the IAM.

def forward(self, inputs, rnn_hxs, masks):
    hidden_conv = self.cnn(inputs / 255.0)

    fnn_out = self.fnn(hidden_conv)
    inf_hidden = self.attention(hidden_conv, rnn_hxs)
    rnn_out, rnn_hxs = self._forward_gru(inf_hidden, rnn_hxs, masks)

    x = torch.cat((rnn_out,fnn_out), 1)

    hidden_critic = self.critic(x) 
    hidden_actor = self.actor(x)

    return hidden_critic, hidden_actor, rnn_hxs


Finally, to modify the Gym Atari environment to flickering Atari, in main.py, the following is added to the main loop:


if args.flicker:
    prob_flicker = np.random.uniform(0, 1, (obs.shape[0],))
    obs[prob_flicker > 0.5] = 0


This sets each timestep's observation to all zeros with a probability of 0.5.
