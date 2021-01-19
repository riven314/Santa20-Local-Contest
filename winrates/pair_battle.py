"""
set up N battles between 2 agents, compute overall winrate
"""
import os
from kaggle_environments import make

from winrates.utils import GameOutcome, timeit, get_logger

logger = get_logger(__name__)


@timeit
def pair_agent_battle(left_agent: str, right_agent: str, battle_n = 100) -> dict:
    assert os.path.isfile(left_agent)
    assert os.path.isfile(right_agent)
    
    # simulate games and parse stats + outcome
    left_wins = 0
    right_wins = 0
    stats_dict = {}
    logger.info(f'left agent: {left_agent} v.s. right_agent: {right_agent}, battle_n: {battle_n}')

    # TODO: multiprocessing for speedup
    for n in range(battle_n):
        signature = f'[GAME {n + 1}/{battle_n}]'
        logger.info(f'{signature} game starting')
        # TODO: needa shuffle left_agent, right_agent between games coz position may have edge 
        game_stats, game_outcome = _run_one_game(left_agent, right_agent)
        if game_outcome == GameOutcome.TIE.value:
            outcome_str = GameOutcome.TIE.name
        elif game_outcome == GameOutcome.LEFT_WIN.value:
            outcome_str = GameOutcome.LEFT_WIN.name
            left_wins += 1
        else:
            outcome_str = GameOutcome.RIGHT_WIN.name
            right_wins += 1
        stats_dict[n] = game_stats
        logger.info(f'{signature} outcome: {outcome_str}')
    
    # compute overall outcome
    left_winrate = left_wins / battle_n
    right_winrate = right_wins / battle_n
    if left_winrate == right_winrate:
        logger.info('Overall result: TIE')
    elif left_winrate > right_winrate:
        logger.info(f'Overall result: {left_agent} WINS ({left_winrate:.2f})')
    else:
        logger.info(f'Overall result: {right_agent} WINS ({right_winrate:.2f})')

    return left_winrate, stats_dict


@timeit
def _run_one_game(left_agent: str, right_agent: str):
    one_game = make('mab', debug = True)
    one_game.reset()
    one_game.run([left_agent, right_agent])

    # massage time-series stats of the game
    num_steps = len(one_game.steps) - 1
    left_actions = [None] * num_steps
    right_actions = [None] * num_steps
    left_rewards = [None] * num_steps # cumulative rewards at time t
    right_rewards = [None] * num_steps # cumulative rewards at time t
    thresholds = [None] * num_steps
    for i, step in enumerate(one_game.steps[1:]):
        left_env, right_env = step
        left_actions[i], right_actions[i] = left_env['action'], right_env['action']
        left_rewards[i], right_rewards[i] = left_env['reward'], right_env['reward']
        thresholds[i] = left_env['observation']['thresholds']
    # needa book-keep left_agent, right_agent coz left_agent, right_agent may shuffle between games
    stats = {
        'thresholds': thresholds,
        'left_agent': left_agent, 'right_agent': right_agent,
        'left_actions': left_actions, 'right_actions': right_actions,
        'left_rewards': left_rewards, 'right_rewards': right_rewards,
    }

    # get game outcome
    left_total_reward, right_total_reward = left_rewards[-1], right_rewards[-1]
    if left_total_reward == right_total_reward:
        outcome = GameOutcome.TIE.value
    elif left_total_reward > right_total_reward:
        outcome = GameOutcome.LEFT_WIN.value
    else:
        outcome = GameOutcome.RIGHT_WIN.value

    return stats, outcome


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'battle between 2 agents')
    parser.add_argument('--left_agent', type = str, required = True)
    parser.add_argument('--right_agent', type = str, required = True)
    parser.add_argument('-n', type = int, default = 100)
    args = parser.parse_args()

    left_winrate, stats_dict = pair_agent_battle(args.left_agent, args.right_agent, args.n)