
def global_knowledge(agents):
    # Write code to accumulate the knowledge from the agents into a common repository for
    # shared knowledge between agents
    pass

def trace_agent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):  ## CHANGE TO CORRECT PROGRAM
        action = old_program(percept)
        print('{} perceives {} and does {}'.format(agent, percept, action))
        return action
    agent.program = new_program
    return agent



