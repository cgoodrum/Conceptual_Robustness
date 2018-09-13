
import yaml
import networkx as nx
import TSP_Network as TSP_Network
import random as rd
#rd.seed(1)
import utils as ut
import matplotlib.pyplot as plt
import collections
import numpy as np



class Specification(object):
    defaults = {

    }

    def __init__(self, filepath):
        self.filepath = filepath
        self.spec = {}
        self.parameters = {}
        self.scenarios = {}
        self.network = None
        self.agents = {}
        self.T = 0
        self.num_trials = 1

    def parse(self):
        with open(self.filepath, 'r') as specfile:
            spec = yaml.load(specfile)
        self.spec = spec


    ######################################### INITIALIZATION #####################################################

    def set_network(self):
        network_spec = self.spec.get('network', False)
        node_spec = network_spec.get('nodes', False)
        edge_spec = network_spec.get('edges', False)

        if network_spec:
            loadfile = network_spec.get('load', False)
            if loadfile: # If filepath presented to load network, do so. Otherwise, create network.
                self.network = nx.read_yaml(loadfile)
            else:
                self.network = TSP_Network.create_network(node_spec,edge_spec)

            return self.network

    def set_agents(self, environment):

        agent_spec = self.spec.get('agents')
        mav_spec = agent_spec.get('mavericks')
        fol_spec = agent_spec.get('followers')
        HE_spec = agent_spec.get('HE_agents')
        j = 0
        agents = {}

        if mav_spec:
            num_mavs = mav_spec['number']

            mavericks = {}
            for i in range(num_mavs):
                mavericks[i] = Maverick(name="Mav_" + str(i), location = self.get_random_location(self.network),
                                        environment=environment)
                agents[j] = mavericks[i]
                j += 1

        if fol_spec:
            num_fols = fol_spec['number']
            followers = {}
            for i in range(num_fols):
                followers[i] = Follower(name="Fol_" + str(i), location = self.get_random_location(self.network),
                                        environment=environment)
                agents[j] = followers[i]
                j += 1

        if HE_spec:
            num_HE = HE_spec['number']
            HE_agents = {}
            for i in range(num_HE):
                HE_agents[i] = Hill_Agent(name="HE_" + str(i), location = self.get_random_location(self.network),
                                          environment=environment)
                agents[j] = HE_agents[i]
                j += 1

        return agents

    def get_random_location(self, network):
        location = np.random.randint(nx.number_of_nodes(network))
        return location

    def set_T(self):
        # Get the total time horizon
        time_spec = self.spec.get('time')
        if time_spec:
            T = time_spec.get('T')
            self.T = T
        else:
            raise NotImplementedError
        return T

    def set_trials(self):
        trial_spec = self.spec.get('trials')
        if trial_spec:
            num_trials = trial_spec.get('num_runs')
            self.num_trials = num_trials
        else:
            raise NotImplementedError
        return num_trials

    def save_yaml_file(self, results, filepath):
        with open(filepath, 'w') as savefile:
            yaml.dump(results, savefile, default_flow_style=False)

    def run(self):
        results = [1 for i in range(10)]
        settings = self.spec.get('settings', {})
        save = settings.get('save_results', False)
        if save:
            self.save_yaml_file(results, save)

    ###############################################################################

class Environment(object):

    def __init__(self, network=nx.DiGraph(), T=10):
        self.network = network
        self.T = T
        self.current_time = 0

    def tick(self):
        self.current_time += 1



class Agent(object):
    # Perceive is how the agent views the environment, and returns percepts,
    # Interpret is how the percepts are used to infer the agents move (the decision),
    # Act updates tyhe agent in the environment again

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, name=None, location=0, environment=Environment(network=nx.DiGraph(),T=10)):
        self.name = name
        self.location = location
        self.alive = True
        self.performance = 0
        self.percept = {}
        self.action = {}
        self.interpretation = {}
        self.environment = environment
        self.history = {}

    def perceive(self):
        raise NotImplementedError

    def interpret(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

    def get_percepts(self, location, environment):
        raise NotImplementedError


class Hill_Agent(Agent):

    def __init__(self, **kwargs):
        Agent.__init__(self, **kwargs)


    def perceive(self):
        perception = self.get_percepts(self.location, self.environment.network)
        self.percept = perception
        return perception

    def interpret(self):
        # Always follow the shortest path
        if self.percept:
            min_length = min(self.percept.values())
            for k, v in self.percept.items():
                if v == min_length:
                    sel_edge = (k,v)
                    break

            self.interpretation[self.environment.current_time] = sel_edge
            return sel_edge
        else:
            self.alive = False

        # sel_path, sel_dist = rd.choice(list(self.percept.items()))

    def act(self):
        self.history[self.environment.current_time] = self.location
        if self.percept:
            decision = self.interpretation[self.environment.current_time]
            self.location = decision[0][1]
            self.performance += decision[1]
        else:
            decision = self.location
        return decision


    def get_percepts(self, location, environment):
        # Overwrite the base class get_percepts function to update with hill-agent-based rules
        # For a given location (node) in the environment, return the neighbor paths and associated lengths
        possible_paths = self.get_path_information(location=location, environment=environment)
        return possible_paths

    def get_path_information(self, location, environment):
        # path_information = self.get_paths(location,environment)
        path_information = self.get_path_distances(location, environment)
        return path_information

    def get_paths(self, location, environment):
        # Use if agents should only be able to see the number of paths, not the distances associated.
        possible_paths = environment.neighbors(location)
        return possible_paths

    def get_path_distances(self, location, environment):
        # Use if agents should be able to see both paths and distances.
        possible_paths = environment.edges(location)
        distances = nx.get_edge_attributes(environment, 'length')
        remove_edges = []
        if self.history:
            for edge in possible_paths:
                for visited_node in self.history.values():
                    if edge[1] == visited_node:
                        remove_edges.append(edge)
            possible_paths = [edge for edge in possible_paths if edge not in remove_edges]
            location_edges = {k:v for k,v in distances.items() if k in possible_paths}
        else:

            location_edges = {k: v for k, v in distances.items() if k in possible_paths}

        return location_edges


class Maverick(Agent):

    def __init__(self, **kwargs):
        Agent.__init__(self, **kwargs)

    def perceive(self):
        # Interface between the environment and what the agent perceives.
        perception = self.get_percepts(self.location)
        self.percept = perception
        return perception

    def interpret(self):
        # Function that defines how the agent should handle what it sees, and decides what to do.
        pass

        #interpretation = self.interpret(self.percept)
        #self.interpretation = interpretation
        #return interpretation

    def act(self, interpretation):
        # Update the action in the environment
        pass

    def get_percepts(self, location, environment):
        # Overwrite the base class get_percepts function to update with maverick-based rules
        # For a given location (node) in the environment, return the neighbor paths and associated lengths
        possible_paths = self.get_path_information(location=location, environment=environment)

        return possible_paths

    def get_path_information(self, location, environment):
        #path_information = self.get_paths(location,environment)
        path_information = self.get_path_distances(location,environment)
        return path_information

    def get_paths(self, location, environment):
        # Use if agents should only be able to see the number of paths, not the distances associated.
        possible_paths = environment.neighbors(location)
        return possible_paths

    def get_path_distances(self, location, environment):
        # Use if agents should be able to see both paths and distances.
        possible_paths = environment.edges(location)
        distances = nx.get_edge_attributes(environment,'length')
        location_edges = {k:v for k,v in distances.items() if k in possible_paths}
        return location_edges

class Follower(Agent):

    def __init__(self, **kwargs):
        Agent.__init__(self, **kwargs)

    def perceive(self):
        percept = self.get_percepts(self.location)
        return percept

    def interpret(self, percept):
        action = percept
        return action

    def act(self, action):
        # update the action in the environment
        pass

    def get_percepts(self, location, environment):
        # Overwrite the base class get_percepts function to update with follower-based rules
        pass








def main():

    ##Initialize
    yamlpath = 'C:\\Users\\cgoodrum\\Documents\\Research\\Conceptual_Robustness\\template.yml'
    specification = Specification(yamlpath)
    specification.parse()
    time_horizon = specification.set_T()
    num_trials = specification.set_trials()

    for t in range(num_trials):
        network = specification.set_network()
        environment = Environment(network=network, T=time_horizon)
        agents = specification.set_agents(environment=environment)
        while environment.current_time <= environment.T:
            for agent_num, agent in agents.items():
                if agent.alive:
                    agent.perceive()
                    agent.interpret()
                    agent.act()
                else:
                    pass
            environment.tick()

        for agent_num, agent in agents.items():
            print("Trial " + str(t), "  ", agent.name, "  ", agent.history.values(), "  ", agent.performance)
        #print("")
        specification.run()

    TSP_Network.draw_network(environment.network)
    #plt.show()


if __name__ == '__main__':
    main()

