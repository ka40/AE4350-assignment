from visualization import plot_util as pu
import matplotlib as mpl
# mpl.style.use('seaborn')
import matplotlib.pyplot as plt
import matplotlib

#Test
# LOG_DIRS = 'IAM succesful my replication/log_test'
# results = pu.load_results(LOG_DIRS)

# fig1, ax1 = pu.plot_results(results, average_group=True,
#                       split_fn=lambda _:
# '',
#                       shaded_std=True,shaded_err=True,
#                       legend_outside=True, xlabel='Steps',
#                       ylabel='Average Reward')

# plt.show()
# end test
selected_env = 'b_f_lr'
if selected_env == 'ware':
    color = ['blue', 'red', 'black', 'chartreuse']
else:
    # color = ['red','green','blue']
    color = ['red', 'firebrick', 'green', 'chartreuse', 'black', 'blue',
              'royalblue']

# # GROUPED UP
# # LOG_DIRS = 'results-taylan/ware/fnnrnnp/results/1'
if selected_env=='ware':
    LOG_DIRS = 'IAM succesful my replication/logs/warehouse/'
elif selected_env=='b_no_f':
    LOG_DIRS = 'IAM succesful my replication/logs/breakout_no_flicker_long/'
elif selected_env=='b_f':
    LOG_DIRS = 'IAM succesful my replication/logs/breakout_flicker_long/'
elif selected_env=='b_f_lr':
    LOG_DIRS = 'IAM succesful my replication/logs/breakout_flicker_lr/'

results = pu.load_results(LOG_DIRS)


# fig = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)
# fig = pu.plot_results(results, average_group=True, split_fn=lambda _: '',
#                       shaded_std=False,COLORS=color,
#                       xlabel='Steps', ylabel='Average Reward',legend_outside=True)
#


# if selected_env == 'ware':
#     fig = matplotlib.pyplot.gcf()
#     plt.gcf().subplots_adjust(bottom=0.15,top=0.9,left=0.15,right=0.7)
#     fig.set_size_inches(30, 30)
#     plt.savefig('plot_ware.png', dpi=150)
# elif selected_env == 'breakout':
#     fig = matplotlib.pyplot.gcf()
#     plt.gcf().subplots_adjust(bottom=0.15,top=0.9,left=0.15, right=0.3)
#     fig.set_size_inches(30, 30)
#     plt.savefig('plot_breakout.png', dpi=150)
# else:
#     pass
# LOG_DIRS = 'results-taylan/ware/rnn/results/1_1' # FNN result 8 cores VIR
# LOG_DIRS2 = 'results-taylan/ware/rnn/results/1_8' # FNN result 8 cores VIR
#
# results = pu.load_results(LOG_DIRS)
# results2 = pu.load_results(LOG_DIRS2)
#
fig1, ax1 = pu.plot_results(results, average_group=True,
                      split_fn=lambda _:
'',
                      shaded_std=True,shaded_err=True,
                      legend_outside=False, xlabel='Steps',
                      ylabel='Average Reward', COLORS=color)
# fig2, ax2 = pu.plot_results(results, average_group=True,
#                       split_fn=lambda _:
# '',
#                       shaded_std=True,shaded_err=True,
#                       legend_outside=True, xlabel='Steps',
#                       ylabel='Average Reward')




plt.show()

# # plt.plot(results)
# x = LOG_DIRS.split('/')
# filename = x[1]
# # L = fig.legend()
# # L.get_texts()[0].set_text('FNN (1 core)')

# fig.legend(labels=['FNN (8 core)', 'FNN (1 core)'], bbox_to_anchor=(0.9,0.2), loc="lower right",  bbox_transform=fig.transFigure)
#
# plt.savefig('plot_ware.png', dpi=300, bbox_inches='tight')
# plt.show()


"""Code and log
### IAM
_____________breakout long one____________

code:
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 160000000 --num-steps 8 --lr 0.00025 --log-dir ./log_breakout_no_flicker/run2 --IAM

log:
log_breakout_no_flicker/run2
11 hours 50e6
Updates 781350, num timesteps 50006464, FPS 1288 
 Last 10 training episodes: mean/median reward 342.1/378.5, min/max reward 138.0/423.0

Updates 781360, num timesteps 50007104, FPS 1288 
 Last 10 training episodes: mean/median reward 342.1/378.5, min/max reward 138.0/423.0
___________



### Warehouse
All ware
python main.py --env-name warehouse --num-steps 8 --recurrent-policy --log-dir ./log_w/FNN8-1 ; python main.py --env-name warehouse --num-steps 8 --recurrent-policy --log-dir ./log_w/FNN8-2 ; python main.py --env-name warehouse --num-steps 8 --recurrent-policy --log-dir ./log_w/FNN8-3 ; python main.py --env-name warehouse --num-steps 1 --recurrent-policy --log-dir ./log_w/FNN1-1 ; python main.py --env-name warehouse --num-steps 1 --recurrent-policy --log-dir ./log_w/FNN1-2 ; python main.py --env-name warehouse --num-steps 1 --recurrent-policy --log-dir ./log_w/FNN1-3 ; python main.py --env-name warehouse --num-steps 8 --log-dir ./log_w/GRU-1 ; python main.py --env-name warehouse --num-steps 8 --log-dir ./log_w/GRU-2 ; python main.py --env-name warehouse --num-steps 8 --log-dir ./log_w/GRU-3 ; python main.py --env-name warehouse --num-steps 8 --IAM --log-dir ./log_w/IAM-1 ; python main.py --env-name warehouse --num-steps 8 --IAM --log-dir ./log_w/IAM-2 ; python main.py --env-name warehouse --num-steps 8 --IAM --log-dir ./log_w/IAM-3

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
Run IAM
```bash
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 50007104 --num-steps 8 --lr 0.00025 --log-dir ./log_b/IAM-1 --IAM
'''

### Breakout flicker

Run IAM with correct hyperparameters
```bash
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 50007104 --num-steps 8 --ppo-epoch 3 --lr 0.00025 --log-dir ./log_bf/IAM-1 --IAM --flicker 42667 seconds
```

### tuning learning rate breakout flicker
{10-2, 10−3, 10-4, 10-5}
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 10000000 --num-steps 8 --ppo-epoch 3 --lr 0.01 --log-dir ./log_bf_lr/IAM_lr_001-1 --IAM --flicker 9177 seconds
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0003 --log-dir ./log_bf_lr/IAM_lr_00003-1 --IAM --flicker
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0002 --log-dir ./log_bf_lr/IAM_lr_00002-1 --IAM --flicker
python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0001 --log-dir ./log_bf_lr/IAM_lr_00001-1 --IAM --flicker 17857 seconds

python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0003 --log-dir ./log_bf_lr/IAM_lr_00003-1 --IAM --flicker ; python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0002 --log-dir ./log_bf_lr/IAM_lr_00002-1 --IAM --flicker ; python main.py --env-name BreakoutNoFrameskip-v4 --num-processes 8 --num-env-steps 20000000 --num-steps 8 --ppo-epoch 3 --lr 0.0001 --log-dir ./log_bf_lr/IAM_lr_00001-1 --IAM --flicker



"""


