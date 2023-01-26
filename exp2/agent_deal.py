import matplotlib.pyplot as plt
import numpy as np
COLORS = ['red', 'green', 'blue']
ALGO = ['Bandit Selection', 'Uniform Sample', 'Iteration Fashion']
fig, ax = plt.subplots(1, 4, figsize=(35, 6.4))
FONT_SIZE = 20
for i in range(4):
    ax[i].grid()
    ax[i].set_xlabel('Iteration', fontdict={'size':FONT_SIZE})
    ax[i].set_ylabel('Diversity', fontdict={'size':FONT_SIZE*1.2})
    ax[i].tick_params(labelsize=FONT_SIZE)

for fig_num in range(4):
    ax[fig_num].set_title('N = {}'.format(4 * (fig_num + 1)), fontsize=FONT_SIZE*1.2)
    with open('output_{}_8.txt'.format(4 * (fig_num + 1))) as file:
        lines = file.readlines()
    for i in range(3):
        data  = []
        for j in range(6):
            data.append(np.array(eval(lines[j * 3 + i])))
        data = np.array(data)
        mean_data = data.mean(axis = 0)
        var_data = np.std(data, axis = 0)
        data_min, data_max = mean_data - var_data, mean_data + var_data
        xtick = np.arange(data.shape[1])
        ax[fig_num].fill_between(xtick, data_min, data_max, facecolor=COLORS[i], alpha=0.2)
        ax[fig_num].plot(xtick, mean_data, COLORS[i], label=ALGO[i])
    # import pdb;pdb.set_trace()

handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 1]
ax[1].legend([handles[i] for i in order], [labels[i] for i in order])
leg = ax[1].legend(ncol=3, loc='center left', bbox_to_anchor=(0.25, 1.14), fontsize=FONT_SIZE, columnspacing=1)

for i in range(3):
    leg.get_lines()[i].set_linewidth(5)

plt.savefig('result_agent.png', bbox_inches='tight')
plt.savefig('result_agent.pdf', bbox_inches='tight')