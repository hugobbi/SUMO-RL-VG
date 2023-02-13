import argparse
import os
import sys
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

# FIXED TIME, ONE CONTEXT
def fixed_times(runs):
    env = SumoEnvironment(
        net_file='nets/arterial_grid/arterialGrid.net.xml',
        route_file='nets/arterial_grid/arterialGrid.rou.xml',
        #additional_sumo_cmd='-a /nets/arterial_grid/arterialGrid2Contexts.add.xml',
        use_gui=False,
        num_seconds=10000,
        # delta_time=5,
        fixed_ts=True)

    for run in range(runs):
        env.reset()
        done = {'__all__': False}

        while not done['__all__']:
            s, r, done, info = env.step(None)

        env.save_csv('outputs/arterialGrid/arterialGridFixedTimes', run)
        env.close()


# ACTUATED, ONE CONTEXT
def actuated(runs):
    env = SumoEnvironment(
        net_file='/home/bazzan/pesquisa/traffic/sumoNets/arterialGridNoPedestrians/arterialGridActuatedTLS.net.xml',
        route_file='/home/bazzan/pesquisa/traffic/sumoNets/arterialGridNoPedestrians/arterialGrid.rou.xml',
        additional_sumo_cmd='-a /home/bazzan/pesquisa/traffic/sumoNets/arterialGridNoPedestrians/arterialGrid.add.xml',
        use_gui=True,
        num_seconds=10000,
        # delta_time=5,
        fixed_ts=True)

    for run in range(runs):
        env.reset()
        done = {'__all__': False}

        while not done['__all__']:
            s, r, done, info = env.step(None)

        env.save_csv('outputs/arterialGridNoPedestrians/arterialGridActuated', run)
        env.close()


# FIXED TIMES, WITH 2 CONTEXTS
def twoContextsFixed(runs):
    env = SumoEnvironment(
        net_file='nets/arterial-grid/arterialGrid.net.xml',
        route_file='nets/arterial-grid/arterialGrid.rou.xml',
        additional_sumo_cmd='-a nets/arterial-grid/arterialGrid2Contexts.add.xml',
        use_gui=False,
        num_seconds=25000,
        # delta_time=5,
        fixed_ts=True)

    for run in range(runs):
        env.reset()
        done = {'__all__': False}

        while not done['__all__']:
            s, r, done, info = env.step(None)

        env.save_csv('outputs/arterialGridNoPedestrians/arterialGrid2ContextsFixed', run)
        env.close()


# ACTUATED, 2 CONTEXTS
def twoContextsActuated(runs):
    env = SumoEnvironment(
        net_file='/home/bazzan/pesquisa/traffic/sumoNets/arterialGridNoPedestrians/arterialGridActuatedTLS.net.xml',
        route_file='/home/bazzan/pesquisa/traffic/sumoNets/arterialGridNoPedestrians/arterialGrid.rou.xml',
        additional_sumo_cmd='-a /home/bazzan/pesquisa/traffic/sumoNets/arterialGridNoPedestrians/arterialGrid2Contexts.add.xml',
        use_gui=False,
        num_seconds=20000,
        # delta_time=5,
        fixed_ts=True)

    for run in range(runs):
        env.reset()
        done = {'__all__': False}

        while not done['__all__']:
            s, r, done, info = env.step(None)

        env.save_csv('outputs/arterialGridNoPedestrians/arterialGrid2ContextsActuated', run)
        env.close()

# QL, ONE CONTEXT
def q_learning(runs):
    alpha = 0.1
    gamma = 0.75
    decay = 1.

    env = SumoEnvironment(
        net_file='nets/arterial-grid/arterialGrid.net.xml',
        route_file='nets/arterial-grid/arterialGrid.rou.xml',
        #additional_sumo_cmd='-a /home/bazzan/pesquisa/traffic/sumoNets/arterialGridNoPedestrians/arterialGrid.add.xml',
        # reward_fn="average-speed",
        reward_fn="queue",
        use_gui=True,
        num_seconds=10000,
        min_green=10,
        max_green=50,
        yellow_time=3,
        delta_time=5)

    for run in range(runs):
        initial_states = env.reset()
        if run == 0:
            ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                     state_space=env.observation_space,
                                     action_space=env.action_space,
                                     alpha=alpha,
                                     gamma=gamma,
                                     exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.05,
                                                                        decay=decay)) for ts in env.ts_ids}
        for ts in initial_states.keys():
            ql_agents[ts].state = env.encode(initial_states[ts], ts)

        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)

            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

        env.save_csv('outputs/arterialGrid/arterialGridQL_queue', run)
        env.close()


# QL, 2 CONTEXTS
def twoContexts_q_learning(runs):
    alpha = 0.05
    gamma = 0.95
    decay = 0.995
    env = SumoEnvironment(
        net_file='nets/arterial-grid-2-context/arterialGrid.net.xml',
        route_file='nets/arterial-grid-2-context/arterialGrid.rou.xml',
        additional_sumo_cmd='-a nets/arterial-grid-2-context/arterialGrid2Contexts.add.xml',
        reward_fn="queue",
        use_gui=True,
        num_seconds=25000,
        min_green=10,
        max_green=50,
        yellow_time=3,
        delta_time=5)

    for run in range(runs):
        initial_states = env.reset()
        if run == 0:
            ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                     state_space=env.observation_space,
                                     action_space=env.action_space,
                                     alpha=alpha,
                                     gamma=gamma,
                                     exploration_strategy=EpsilonGreedy(initial_epsilon=1., min_epsilon=0.05,
                                                                        decay=decay)) for ts in env.ts_ids}
        for ts in initial_states.keys():
            ql_agents[ts].state = env.encode(initial_states[ts], ts)

        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)

            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

        env.save_csv('outputs/arterialGridNoPedestrians/arterialGridQL_queue_2contexts', run)
        env.close()


ACTION = 0
Q_VALUES = 1
DISTANCES = 2
OBSERVATIONS = 3

'''
def knn(runs, k):
    alpha = 0.05
    gamma = 0.95
    initial_epsilon = 1
    min_epsilon = 0.05
    decay = 0.995

    env = SumoEnvironment(net_file='nets/arterial-grid/arterialGrid.net.xml',
                          route_file='nets/arterial-grid/arterialGrid.rou.xml',
                          additional_sumo_cmd='-a nets/arterial-grid/arterialGrid2Contexts.add.xml',
                          reward_fn='queue',
                          use_gui=False,
                          num_seconds=25000,
                          min_green=10,
                          max_green=50,
                          yellow_time=3,
                          delta_time=5)

    for run in range(runs):
        initial_states = env.reset()
        if run == 0:
            knn_td_agents = {ts: KNNTDAgent(starting_state=initial_states[ts],
                                            state_space=env.observation_space,
                                            action_space=env.action_space,
                                            k=k,
                                            alpha=alpha,
                                            gamma=gamma,
                                            initial_epsilon=initial_epsilon,
                                            min_epsilon=min_epsilon,
                                            decay=decay) for ts in env.ts_ids}

        for ts in initial_states.keys():
            knn_td_agents[ts].state = tuple(initial_states[ts])

        infos = []
        done = {'__all__': False}
        s = initial_states
        step = 0
        while not done['__all__']:
            actions = {ts: knn_td_agents[ts].act() for ts in knn_td_agents.keys()}

            step += 1

            s, r, done, info = env.step(action={ts: actions[ts][ACTION] for ts in knn_td_agents.keys()})

            for agent_id in s.keys():
                knn_td_agents[agent_id].learn(next_state=s[agent_id], reward=r[agent_id])

        env.save_csv('outputs/arterialGridNoPedestrians/arterialGrid-2ContextsKNN-k' + str(k), run)
        run += 1
        env.close()


def knn_with_transfer(runs, k, transfer_distance_threshold):

    alpha = 0.05
    gamma = 0.95
    initial_epsilon = 1
    min_epsilon = 0.05
    decay = 0.995

    env = SumoEnvironment(net_file='nets/arterial-grid/arterialGrid.net.xml',
                          route_file='nets/arterial-grid/arterialGrid.rou.xml',
                          additional_sumo_cmd='-a nets/arterial-grid/arterialGrid2Contexts.add.xml',
                          reward_fn='queue',
                          use_gui=False,
                          num_seconds=25000,
                          min_green=10,
                          max_green=50,
                          yellow_time=3,
                          delta_time=5)

    for run in range(runs):
        initial_states = env.reset()
        if run == 0:
            knn_td_agents = {ts: KNNTransferAgent(starting_state=initial_states[ts],
                                                  state_space=env.observation_space,
                                                  action_space=env.action_space,
                                                  k=k,
                                                  transfer_distance_threshold=transfer_distance_threshold,
                                                  alpha=alpha,
                                                  gamma=gamma,
                                                  initial_epsilon=initial_epsilon,
                                                  min_epsilon=min_epsilon,
                                                  decay=decay) for ts in env.ts_ids}

        for ts in initial_states.keys():
            knn_td_agents[ts].state = tuple(initial_states[ts])

        infos = []
        done = {'__all__': False}
        s = initial_states
        step = 0
        while not done['__all__']:
            actions = {ts: knn_td_agents[ts].act() for ts in knn_td_agents.keys()}

            step += 1

            s, r, done, info = env.step(action={ts: actions[ts] for ts in knn_td_agents.keys()})

            for agent_id in s.keys():
                knn_td_agents[agent_id].learn(next_state=s[agent_id], reward=r[agent_id])

            for agent_id in s.keys():
                if knn_td_agents[agent_id].request_experiences:
                    print('Experience requested')
                    for other_agent in [agent for agent in s.keys() if agent != agent_id]:
                        experiences = knn_td_agents[other_agent].send_experiences(knn_td_agents[agent_id].state)
                        knn_td_agents[agent_id].receive_experiences(experiences)

        env.save_csv('outputs/arterialGridNoPedestrians/arterialGrid-2ContextsKNNTransfer-k' + str(k) + '-threshold' + str(transfer_distance_threshold), run)
        run += 1
        env.close()
'''

if __name__ == '__main__':
    # twoContextsFixed(10)
    # twoContexts_q_learning(5)
    # knn(runs=5, k=10)
    # knn(runs=5, k=50)
    # knn(runs=5, k=200)
    #knn_with_transfer(runs=1, k=10, transfer_distance_threshold=0.01)
    #knn_with_transfer(runs=1, k=10, transfer_distance_threshold=0.001)
    #knn_with_transfer(runs=1, k=10, transfer_distance_threshold=0.1)
    #knn_with_transfer(runs=1, k=10, transfer_distance_threshold=0.05)
    twoContexts_q_learning(1)
