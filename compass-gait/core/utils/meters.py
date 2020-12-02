import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from jax.numpy.linalg import norm as jnorm

class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        """Computes and stores the average and current value.
        
        Params:
            name: Name of variable that we are keeping track of.
            fmt: Format for printing data.
        """
        
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all of the intermediate values and averages."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average value.
        
        Params:
            val: Value being appended.
            n: Number of items that were used to compute val.
        """

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Saver:
    def __init__(self, args, consts_df_path=None, box_df_path=None):

        self.args = args
        self.losses, self.epochs = [], []
        self.safe_pct, self.unsafe_pct, self.cnt_pct, self.dis_pct = ([] for _ in range(4))
        self.safe_hvals, self.unsafe_hvals = None, None
        self.consts_df_path, self.box_df_path = consts_df_path, box_df_path
        self.safe_dv, self.unsafe_dv, self.cnt_dv, self.dis_dv = ([] for _ in range(4))

        self.root = os.path.join(args.results_dir, 'training')
        os.makedirs(self.root, exist_ok=True)

    def plot(self):
        """Plot constraint satisfaction, training loss, and boxplot."""

        df = self.to_dataframe()
        dual_df = self.to_dual_df()

        sns.set_style('whitegrid')
        sns.set(font_scale=1.2)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(14, 5))
        self.plot_consts(ax1, df)
        self.plot_loss(ax2, df)
        self.plot_dual_vars(ax3, dual_df)
        self.plot_boxplot(ax4)

        fname = os.path.join(self.args.results_dir, 'train.png')
        plt.savefig(fname)
        plt.tight_layout()
        plt.close(fig)

    def update(self, epoch, loss, constraints, safe_hvals, unsafe_hvals, dual_vars=None):
        """Update training statistics and lists."""

        self.epochs.append(epoch)
        self.losses.append(float(loss))
        self.safe_pct.append(float(constraints['safe']))
        self.unsafe_pct.append(float(constraints['unsafe']))
        self.cnt_pct.append(float(constraints['cnt']))
        self.dis_pct.append(float(constraints['disc']))
        self.safe_hvals = safe_hvals
        self.unsafe_hvals = unsafe_hvals

        if dual_vars is not None:

            def to_float(name):
                return float(jnorm(dual_vars[f'λ_{name}'], ord=2))

            self.safe_dv.append(to_float('safe'))
            self.unsafe_dv.append(to_float('unsafe'))
            self.cnt_dv.append(to_float('cnt'))
            self.dis_dv.append(to_float('dis'))

    def to_dataframe(self):
        """Convert saved data to dataframe (for plotting)"""

        if self.consts_df_path is None:
            data = list(zip(
                self.epochs, self.losses,
                self.safe_pct, self.unsafe_pct, self.cnt_pct, self.dis_pct
            ))
            cols = ['Epochs', 'Loss', 'Safe_pct', 'Unsafe_pct', 'Cnt_pct', 'Dis_pct']
            df = pd.DataFrame(data, columns=cols)

            with open(os.path.join(self.root, 'constraints.pd'), 'wb') as fp:
                pickle.dump(df, fp)
        else:
            df = self.open_saved_df(self.consts_df_path)

        return df

    def to_dual_df(self):

        data = list(zip(self.epochs, self.safe_dv, self.unsafe_dv, 
            self.cnt_dv, self.dis_dv))
        cols = ['Epoch', 'λ_safe', 'λ_unsafe', 'λ_cnt', 'λ_dis']
        df = pd.DataFrame(data, columns=cols)

        return df

    def plot_loss(self, ax, df):
        kwargs = {'x': 'Epochs', 'linewidth': 3, 'data': df, 'ax': ax}
        sns.lineplot(y='Loss', **kwargs)
        ax.set_yscale('log')
        ax.set_title('Training loss')
        
    def plot_consts(self, ax, df):

        kwargs = {'x': 'Epochs', 'linewidth': 3, 'data': df, 'ax': ax}
        sns.lineplot(y='Safe_pct', label='Safe (eq. 9a)', **kwargs)
        sns.lineplot(y='Unsafe_pct', label='Unsafe (eq. 9b)', **kwargs)
        sns.lineplot(y='Cnt_pct', label='Continuous (eq. 9d)', **kwargs)
        sns.lineplot(y='Dis_pct', label='Discrete (eq. 9f)', **kwargs)
        ax.set_ylabel('Constraint satisfaction percentage')
        ax.set_title('Constraint satisfaction')
        ax.legend(loc='lower right')

    def plot_dual_vars(self, ax, df):

        kwargs = {'x': 'Epoch', 'linewidth': 3, 'data': df, 'ax': ax}
        sns.lineplot(y='λ_safe', label='λ_safe', **kwargs)
        sns.lineplot(y='λ_unsafe', label='λ_unsafe', **kwargs)
        sns.lineplot(y='λ_cnt', label='λ_cnt', **kwargs)
        sns.lineplot(y='λ_dis', label='λ_dis', **kwargs)
        ax.set_ylabel('Values of dual variables')
        ax.set_title('Dual variables')
        ax.legend()

    def plot_boxplot(self, ax):

        def make_df(vals, safe=True):
            truth = 'safe' if safe is True else 'unsafe'
            data = list(zip(vals, [truth for _ in range(len(vals))]))
            return pd.DataFrame(data, columns=['h(x)', 'States'])

        if self.box_df_path is None:
            
            safe_df = make_df(self.safe_hvals, safe=True)
            unsafe_df = make_df(self.unsafe_hvals, safe=False)
            df = pd.concat([safe_df, unsafe_df], ignore_index=True)

            with open(os.path.join(self.root, 'boxplot.pd'), 'wb') as fp:
                pickle.dump(df, fp)

        else:
            df = self.open_saved_df(self.box_df_path)

        sns.boxplot(x='States', y='h(x)', data=df)
        ax.set_title('State separation')
        
    @staticmethod
    def open_saved_df(path):
        with open(path, 'rb') as fp:
            df = pickle.load(fp)

        return df