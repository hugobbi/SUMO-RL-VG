import argparse as ap
import os
import sys
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
parser.add_argument("-f", "--vg-file", dest="vg_file", 
                             help="Path to pickle file with information of virtual graph", default=None)
args = parser.parse_args()

# QL
def q_learning(runs):
    print("Running simulation with QL and no VG")

    alpha = 0.05
    gamma = 0.95
    decay = 0.995
    env = SumoEnvironment(
        net_file='nets/diamond_tls/DiamondTLs.net.xml',
        route_file='nets/diamond_tls/DiamondTLs.flow_099.rou.xml',
        # reward_fn="average-speed",
        reward_fn="queue",
        use_gui=False,
        num_seconds=15000,
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


        env.save_csv('outputs/diamond_tls/ql_no_vg/diamond_tls_ql' + str(random.randint(1, 1000)), run)
        env.close()

# QL + VG
def q_learning_vg(runs, vg_neighbors_dict_file):
    print("Running simulation with QL and VG")
    print("Reading graph neighbors dictionary from pickle file...")
    with open(vg_neighbors_dict_file, "rb") as vg_dict_pickle: # args.vg_file
        vg_neighbors_dict = pickle.load(vg_dict_pickle, encoding="bytes")
    print(f"VG neighbors loaded, size of dict = {len(vg_neighbors_dict.keys())}")

    alpha = 0.05
    gamma = 0.95
    decay = 0.995
    env = SumoEnvironment(
        net_file='nets/diamond_tls/DiamondTLs.net.xml',
        route_file='nets/diamond_tls/DiamondTLs.flow_099.rou.xml',
        # reward_fn="average-speed",
        reward_fn="queue",
        use_gui=False,
        num_seconds=15000,
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

        agents_in_vg = sorted(vg_neighbors_dict.keys()) # agents that have a connection in the vg and will learn from this connection
        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)

            current_step = env.sim_step
            print(f"{current_step=}")
            for agent_id in agents_in_vg:
                vg_neighbors_in_current_step = get_graph_neighbors_interval(vg_neighbors_dict[agent_id], current_step)
                print(f"{agent_id}: {vg_neighbors_in_current_step}")
                for vg_neighbor_id in vg_neighbors_in_current_step:
                    # agent updates its q-table with vg information
                    ql_agents[agent_id].update_q_table(ql_agents[vg_neighbor_id].state, ql_agents[vg_neighbor_id].action, reward=r[vg_neighbor_id], next_state=env.encode(s[vg_neighbor_id], vg_neighbor_id))

            for agent_id in s.keys():
                # agent learns with its own information
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

        folder_name_graph_size = vg_neighbors_dict_file[-6:-4] # adds sim files to new folder separating by graph size
        env.save_csv(f'outputs/diamond_tls/ql_vg/{folder_name_graph_size}/diamond_tls_ql_vg' + str(random.randint(1, 1000)), run)
        env.close()

def get_graph_neighbors_interval(graph_neighbors: dict, current_step: int) -> list:
        number_of_intervals = len(graph_neighbors)
        i = 0
        for interval in graph_neighbors:
            if i == number_of_intervals - 1:
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
    if args.vg_file is None:
        q_learning(1)
    else:
        q_learning_vg(1, args.vg_file)
