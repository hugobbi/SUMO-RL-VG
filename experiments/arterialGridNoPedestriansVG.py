import argparse as ap
import os
import sys
import pandas as pd
import pickle
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

parser = ap.ArgumentParser()
parser.add_argument("-f", "--vg_file", 
                             help="Path to pickle file with information of virtual graph")
args = parser.parse_args()

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

        env.save_csv('outputs/arterialGridNoPedestrians/arterialGrid2ContextsFixed_' + str(random.randint(1, 1000)), run)
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
        additional_sumo_cmd='-a nets/arterial-grid/arterialGrid.add.xml',
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

# HENRIQUE: VC NAO DEVIA TER ALTERADO O QL COM 2 CONTEXTS MAS SIM CRIADO UM NOVO BLOCO AQUI ! EH PRECISO INSERIR O BLOCO DO QL COM 2 CONTEXTOS NOVAMENTE AQUI !
# FEITO!

# QL, 2 CONTEXTS
def twoContexts_q_learning(runs):
    alpha = 0.05
    gamma = 0.95
    decay = 0.995
    env = SumoEnvironment(
        net_file='nets/arterial-grid/arterialGrid.net.xml',
        route_file='nets/arterial-grid/arterialGrid.rou.xml',
        additional_sumo_cmd='-a nets/arterial-grid/arterialGrid2Contexts.add.xml',
        reward_fn="queue",
        use_gui=False,
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
     
# QL, 2 CONTEXTS + VG
def twoContexts_q_learning_VG(runs, vg_neighbors_dict):
    print(vg_neighbors_dict)
    alpha = 0.05
    gamma = 0.95
    decay = 0.995
    env = SumoEnvironment(
        net_file='nets/arterial-grid-2-context/arterialGrid.net.xml',
        route_file='nets/arterial-grid-2-context/arterialGrid.rou.xml',
        additional_sumo_cmd='-a nets/arterial-grid-2-context/arterialGrid2Contexts.add.xml',
        reward_fn="queue",
        use_gui=False,
        num_seconds=20000,
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

            current_step = env.sim_step
            print(f"{current_step=}")
            for agent_id in s.keys():
                vg_neighbors_in_current_step = get_graph_neighbors_interval(vg_neighbors_dict[agent_id], current_step)
                print(f"{agent_id}: {vg_neighbors_in_current_step}")
                for vg_neighbor_id in vg_neighbors_in_current_step:
                    # agent updates its q-table with vg information
                    ql_agents[agent_id].update_q_table(ql_agents[vg_neighbor_id].state, ql_agents[vg_neighbor_id].action, reward=r[vg_neighbor_id], next_state=env.encode(s[vg_neighbor_id], vg_neighbor_id))

            for agent_id in s.keys():
                # agent learns with its own information
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

        env.save_csv('outputs/arterialGridNoPedestrians/arterialGridQLVG_queue_2contexts_' + str(random.randint(1, 1000)), run)
        env.close()

def get_graph_neighbors_interval(graph_neighbors: dict, current_step: int) -> list:
        number_of_intervals = len(graph_neighbors)
        i = 0
        for interval in graph_neighbors:
            if i == number_of_intervals-1:
                if interval[0] <= current_step <= interval[1]:
                    return graph_neighbors[interval]
            else:
                if interval[0] < current_step <= interval[1]:
                    return graph_neighbors[interval]
            i += 1
        return []

ACTION = 0
Q_VALUES = 1
DISTANCES = 2
OBSERVATIONS = 3



if __name__ == '__main__':
    print("Reading graph neighbors dictionary from pickle file...")
    with open(f"{args.vg_file}", "rb") as vg_dict_pickle:
        vg_neighbors_dict = pickle.load(vg_dict_pickle, encoding="bytes")
    # twoContextsFixed(1)
    # twoContexts_q_learning(1)
    twoContexts_q_learning_VG(1, vg_neighbors_dict)
