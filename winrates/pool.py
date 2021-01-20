import os

import numpy as np
import matplotlib.pyplot as plt

from winrates.utils import timeit, get_logger
from winrates.pair import pair_agent_battle

logger = get_logger(__name__)


def pool_agent_battle(pool_dir, battle_n = 100):
    agents = [os.path.join(pool_dir, fn) for fn in os.listdir(pool_dir)]
    winrate_mat = _compute_winrate_matrix(agents, battle_n)
    agent_labels = [os.path.basename(agent) for agent in agents]
    fig, ax = plot_matrix(agent_labels, winrate_mat)
    fig.show()
    return fig, ax


@timeit
def _compute_winrate_matrix(agents, battle_n):
    # init winrate matrix
    agents_n = len(agents)
    # mat[i, j]: winrate of agent i against agent j
    winrate_mat = np.zeros((agents_n, agents_n))

    # TODO @Alex: try to speed up by multiprocessing
    for i in range(agents_n):
        for j in range(agents_n):
            if j > i:
                continue
            left_agent, right_agent = agents[i], agents[j]
            left_name, right_name = os.path.basename(left_agent), os.path.basename(right_agent)
            logger.info(f'\n[Battle] LEFT: {left_name} v.s. RIGHT: {right_name}\n')
            left_winrate, _ = pair_agent_battle(
                left_agent = left_agent, 
                right_agent = right_agent, 
                battle_n = battle_n)
            winrate_mat[i, j] = left_winrate

    # complete the whole winrate matrix
    upper_mat = winrate_mat.T.copy()
    np.fill_diagonal(upper_mat, 0.)
    upper_mat = 1 - upper_mat
    upper_mat = upper_mat - np.tri(*upper_mat.shape)
    winrate_mat = winrate_mat + upper_mat

    return winrate_mat


def plot_matrix(agent_labels, matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # create labels along x, y axis
    ax.set_yticks(range(len(agent_labels)))
    ax.set_xticks(range(len(agent_labels)))
    ax.set_xticklabels(agent_labels)
    ax.set_yticklabels(agent_labels)
    plt.setp(ax.get_xticklabels(), rotation = 45,
             ha = 'right', rotation_mode = 'anchor')

    # annotate winrate on each cell
    for i in range(len(agent_labels)):
        for j in range(len(agent_labels)):
            score = round(matrix[i, j], 2)
            text = ax.text(j, i, score,
                           ha = 'center',
                           va = 'center',
                           color = 'w')

    ax.set_title("Winrate Matrix for a Pool of Agents")
    fig.tight_layout()
    return fig, ax


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'battle between a pool of agents')
    parser.add_argument('--agent_dir', type = str, required = True)
    parser.add_argument('--output_plot', type = str, default = None)
    parser.add_argument('-n', type = int, default = 100)
    args = parser.parse_args()

    fig, ax = pool_agent_battle(args.agent_dir, args.n)
    if args.output_plot is not None:
        logger.info(f'Winrate matrix plot written: {args.output_plot}')
        fig.savefig(args.output_plot)