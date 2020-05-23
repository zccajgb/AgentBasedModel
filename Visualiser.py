import pandas as pd
from NanoparticleModel import Nanoparticle
from ReceptorModel import Receptor
import matplotlib.pyplot as plt


def visualiser(model):
    """Nanoparticles"""
    agent_nanoparticles_dictionary = {'Points': 'Bound'}
    for agent in model.agents:
        if isinstance(agent, Nanoparticle):
            agent_nanoparticles_dictionary[tuple(agent.position.tolist())] = agent.bound
    nanoparticles = [agent for agent in model.agents if isinstance(agent, Nanoparticle)]
    x = [i.position[0] for i in nanoparticles]
    y = [i.position[1] for i in nanoparticles]
    z = [i.position[2] for i in nanoparticles]
    bound_nanoparticles = [str(i.bound) for i in nanoparticles]
    dictionary = {'x': x, 'y': y, 'z': z, 'Bound': bound_nanoparticles}
    nanoparticles_df = pd.DataFrame(dictionary)
    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    for position, bound in agent_nanoparticles_dictionary.items():
        if bound is False:
            x1.append(position[0])
            y1.append(position[1])
            z1.append(position[2])
        if bound is True:
            x2.append(position[0])
            y2.append(position[1])
            z2.append(position[2])
    '''Receptors'''
    agent_receptors_dictionary = {'Points': 'Bound'}
    for agent in model.agents:
        if isinstance(agent, Receptor):
            model.agent_receptors_dictionary[tuple(agent.position.tolist())] = agent.bound
    receptors = [agent for agent in model.agents if isinstance(agent, Receptor)]
    x3 = [i.position[0] for i in receptors]
    y3 = [i.position[1] for i in receptors]
    z3 = [i.position[2] for i in receptors]
    bound_receptors = [str(i.bound) for i in receptors]
    dictionary = {'x': x3, 'y': y3, 'z': z3, 'Bound': bound_receptors}
    receptors_df = pd.DataFrame(dictionary)
    x4 = []
    y4 = []
    z4 = []
    x5 = []
    y5 = []
    z5 = []
    for position, bound in agent_receptors_dictionary.items():
        if bound is None:
            x4.append(position[0])
            y4.append(position[1])
            z4.append(position[2])
        elif bound is not (None or 'Bound'):
            x5.append(position[0])
            y5.append(position[1])
            z5.append(position[2])
    '''Ligands'''
    x6 = []
    y6 = []
    z6 = []
    for agent in model.agents:  # loop through agents
        if isinstance(agent, Nanoparticle):
            for i in agent.ligands:
                x6.append(i.position[0])
                y6.append(i.position[1])
                z6.append(i.position[2])
    '''3D'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, y1, z1, c='Red', s=350 * (model.nanoparticle_radius / 50))  # Unbound
    ax.scatter(x2, y2, z2, c='Green', s=350 * (model.nanoparticle_radius / 50))  # Bound
    ax.scatter(x4, y4, z4, c='Red', s=model.receptor_length, marker='1')  # Unbound
    ax.scatter(x5, y5, z5, c='Green', s=model.receptor_length, marker='1')  # Bound
    # ax.scatter(x6, y6, z6, c='Blue', s=model.ligand_length, marker='1')  # Ligands
    ax.set_title('3D')
    ax.set_xlabel('X position (nm)')
    ax.set_ylabel('Y position (nm)')
    ax.set_zlabel('Z position (nm)')
    ax.set_xlim(0, model.dimension)
    ax.set_ylim(0, model.dimension)
    ax.set_zlim(0, model.dimension)
    plt.show()
    '''2D'''
    # plt.scatter(x1, y1, c='Red', s=2 * (model.nanoparticle_radius + model.ligand_length))
    # plt.scatter(x2, y2, c='Green', s=2 * (model.nanoparticle_radius + model.ligand_length))
    # plt.title('2D')
    # plt.xlabel('X Axis')
    # plt.ylabel('Y Axis')
    # plt.xlim(0, model.dimension)
    # plt.ylim(0, model.dimension)
    # plt.show()

