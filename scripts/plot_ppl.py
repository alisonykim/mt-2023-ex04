import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


baseline_f = '../models/baseline/baseline.log'
prenorm_f = '../models/deen_transformer_pre/validations.txt'
postnorm_f = '../models/deen_transformer_post/validations.txt'


def get_baseline_ppl(log_file):
    '''Extract validation perplexities from baseline log file and save values in list'''
    perplexities = []
    with open(log_file, 'r') as f:
        for line in f:
            split_line = line.split('-')
            log = [i.strip() for i in split_line]
            if 'joeynmt.prediction' in log:
                if 'ppl' in log[-1]:
                    results = log[-1].split(',')
                    ppl = results[1].split(':')
                    ppl = float(ppl[1].strip())
                    perplexities.append(ppl)
    return perplexities


def get_ppl(log_file):
    '''Extract validation perplexities from pre- or postnorm log file and save values in list'''
    perplexities = []
    steps = []
    with open(log_file, 'r') as f:
        for line in f:
            log = line.split('\t')
            ppl = log[3].split(':')
            ppl = float(ppl[1].strip())
            step = log[0].split(':')
            step = int(step[1].strip())
            perplexities.append(ppl)
            steps.append(step)
    return steps, perplexities


# get lists with baseline, pre- and postnorm validation perplexities
base_ppls = get_baseline_ppl(baseline_f)
steps, pre_ppls = get_ppl(prenorm_f)
steps, post_ppls = get_ppl(postnorm_f)

assert len(base_ppls) == len(pre_ppls) == len(post_ppls) == len(steps)

# initialize dictionary to store all perplexities
ppl_data = {'Validation ppl': steps, 'Baseline': base_ppls, 'Prenorm': pre_ppls, 'Postnorm': post_ppls}

# create dataframe with ppl data
df = pd.DataFrame(data=ppl_data)

# create line chart to plot val perplexities
line_chart = sns.lineplot(data=pd.melt(df, ['Validation ppl']), x='Validation ppl', y='value',
                          hue='variable', palette=['green', 'blue', 'orange'])
line_chart.set(xlabel='Steps', ylabel='Perplexities')
line_chart.set_title('Validation perplexities')
plt.show()

plot_path = '../ppl_plot'

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

line_chart.figure.savefig(plot_path+'/ppl_lineplot.png')