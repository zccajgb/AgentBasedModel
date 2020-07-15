from Master import Master
import numpy as np
import matplotlib.pyplot as plt

### TODO All code below here is repetivive. It needs cleaning up into one funcitonj

def binding_energy():
    print('Binding Energy -------------')
    d = np.linspace(5, 25, 5).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for _ in range(3):
            number_of_seconds = 1  # i.e. 1 hour = 3600 seconds
            model = Master(dimension=1000, binding_energy=int(i), time_unit=10e-3, number_of_receptors=1000, receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100, ligand_length=7, binding_distance=4, receptor_radius=3)
            model.create_receptors()  # 100 nm for receptor
            model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(f'{model.dimension/1000} μm\u00b3 system, {model.binding_energy} binding energy, {model.number_of_nanoparticles} Nanoparticles,\n'
                  f'{model.nanoparticle_radius} nm Nanoparticle Radius, {model.number_of_ligands} Ligands, Ligand length {model.ligand_length} nm,\n'
                  f'{model.number_of_receptors} Receptors, {model.receptor_length} nm Receptor length, {model.binding_distance} Binding distance')
            model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {model.count} reactions')
            print(f'The surface coverage is {model.surface_coverage}')
            variable_finals.append(model.surface_coverage)
            time_data.append(np.array(model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{model.binding_energy} KT binding energy '] = np.array([list(range(0, model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Binding energy (kt)', 'Surface Coverage', d, means, errors)


# binding_energy()


def number_of_receptors():
    print('Number of receptors -------------')
    d = np.linspace(250, 1000, 4).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for _ in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            model = Master(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=int(i),
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=7, binding_distance=4)
            model.create_receptors()  # 100 nm for receptor
            model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(f'{model.dimension / 1000} μm\u00b3 system, {model.binding_energy} binding energy, {model.number_of_nanoparticles} Nanoparticles,\n'
                  f'{model.nanoparticle_radius} nm Nanoparticle Radius, {model.number_of_ligands} Ligands, Ligand length {model.ligand_length} nm,\n'
                  f'{model.number_of_receptors} Receptors, {model.receptor_length} nm Receptor length, {model.binding_distance} Binding distance')
            model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {model.count} reactions')
            print(f'The surface coverage is {model.surface_coverage}')
            variable_finals.append(model.surface_coverage)
            time_data.append(np.array(model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{model.number_of_receptors} receptors'] = np.array([list(range(0, model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Number of receptors', 'Surface Coverage', d, means, errors)


# number_of_receptors()


def receptor_length():
    print('Receptor length -------------')
    d = np.linspace(25, 100, 4).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            model = Master(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=int(i),
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=7, binding_distance=4)
            model.create_receptors()  # 100 nm for receptor
            model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{model.dimension / 1000} μm\u00b3 system, {model.binding_energy} binding energy, {model.number_of_nanoparticles} Nanoparticles,\n'
                f'{model.nanoparticle_radius} nm Nanoparticle Radius, {model.number_of_ligands} Ligands, Ligand length {model.ligand_length} nm,\n'
                f'{model.number_of_receptors} Receptors, {model.receptor_length} nm Receptor length, {model.binding_distance} Binding distance')
            model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {model.count} reactions')
            print(f'The surface coverage is {model.surface_coverage}')
            variable_finals.append(model.surface_coverage)
            time_data.append(np.array(model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{model.receptor_length} nm receptor length'] = np.array(
            [list(range(0, model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Receptor length (nm)', 'Surface Coverage', d, means, errors)


# receptor_length()


def number_of_nanoparticles():
    print('Number of nanoparticles -------------')
    d = np.linspace(60, 180, 3).tolist()
    d.append(190)
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            model = Master(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=int(i), nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=7, binding_distance=4)
            model.create_receptors()  # 100 nm for receptor
            model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{model.dimension / 1000} μm\u00b3 system, {model.binding_energy} binding energy, {model.number_of_nanoparticles} Nanoparticles,\n'
                f'{model.nanoparticle_radius} nm Nanoparticle Radius, {model.number_of_ligands} Ligands, Ligand length {model.ligand_length} nm,\n'
                f'{model.number_of_receptors} Receptors, {model.receptor_length} nm Receptor length, {model.binding_distance} Binding distance')
            model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {model.count} reactions')
            print(f'The surface coverage is {model.surface_coverage}')
            variable_finals.append(model.surface_coverage)
            time_data.append(np.array(model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{model.number_of_nanoparticles} nanoparticles'] = np.array(
            [list(range(0, model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Number of nanoparticles', 'Surface Coverage', d, means, errors)


# number_of_nanoparticles()


def nanoparticle_radius():
    print('Nanoparticle Radius -------------')
    d = np.linspace(10, 50, 5).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            model = Master(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=int(i), number_of_ligands=100,
                               ligand_length=7, binding_distance=4)
            model.create_receptors()  # 100 nm for receptor
            model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{model.dimension / 1000} μm\u00b3 system, {model.binding_energy} binding energy, {model.number_of_nanoparticles} Nanoparticles,\n'
                f'{model.nanoparticle_radius} nm Nanoparticle Radius, {model.number_of_ligands} Ligands, Ligand length {model.ligand_length} nm,\n'
                f'{model.number_of_receptors} Receptors, {model.receptor_length} nm Receptor length, {model.binding_distance} Binding distance')
            model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {model.count} reactions')
            print(f'The surface coverage is {model.surface_coverage}')
            variable_finals.append(model.surface_coverage)
            time_data.append(np.array(model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{model.nanoparticle_radius} nm nanoparticle radius'] = np.array(
            [list(range(0, model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Nanoparticle radius (nm)', 'Surface Coverage', d, means, errors)


# nanoparticle_radius()


def number_of_ligands():
    print('Number of Ligands -------------')
    d = np.linspace(25, 100, 4).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            model = Master(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=int(i),
                               ligand_length=7, binding_distance=4)
            model.create_receptors()  # 100 nm for receptor
            model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{model.dimension / 1000} μm\u00b3 system, {model.binding_energy} binding energy, {model.number_of_nanoparticles} Nanoparticles,\n'
                f'{model.nanoparticle_radius} nm Nanoparticle Radius, {model.number_of_ligands} Ligands, Ligand length {model.ligand_length} nm,\n'
                f'{model.number_of_receptors} Receptors, {model.receptor_length} nm Receptor length, {model.binding_distance} Binding distance')
            model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {model.count} reactions')
            print(f'The surface coverage is {model.surface_coverage}')
            variable_finals.append(model.surface_coverage)
            time_data.append(np.array(model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{model.number_of_ligands} ligands'] = np.array(
            [list(range(0, model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Number of ligands', 'Surface Coverage', d, means, errors)


# number_of_ligands()


def ligand_length():
    print('Ligand Length -------------')
    d = np.linspace(1, 7, 4).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            model = Master(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=int(i), binding_distance=4)
            model.create_receptors()  # 100 nm for receptor
            model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{model.dimension / 1000} μm\u00b3 system, {model.binding_energy} binding energy, {model.number_of_nanoparticles} Nanoparticles,\n'
                f'{model.nanoparticle_radius} nm Nanoparticle Radius, {model.number_of_ligands} Ligands, Ligand length {model.ligand_length} nm,\n'
                f'{model.number_of_receptors} Receptors, {model.receptor_length} nm Receptor length, {model.binding_distance} Binding distance')
            model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {model.count} reactions')
            print(f'The surface coverage is {model.surface_coverage}')
            variable_finals.append(model.surface_coverage)
            time_data.append(np.array(model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{model.ligand_length} nm ligand length'] = np.array(
            [list(range(0, model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Ligand length (nm)', 'Surface Coverage', d, means, errors)


# ligand_length()


def binding_distance():
    print('Binding Distance -------------')
    d = np.linspace(2, 6, 3).tolist()
    data = {}
    means = []
    errors = []
    time_data = []
    for i in d:
        variable_finals = []
        for j in range(3):
            number_of_seconds = 1000  # i.e. 1 hour = 3600 seconds
            model = Master(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=1000,
                               receptor_length=100,
                               number_of_nanoparticles=190, nanoparticle_radius=50, number_of_ligands=100,
                               ligand_length=7, binding_distance=int(i))
            model.create_receptors()  # 100 nm for receptor
            model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
            print(
                f'{model.dimension / 1000} μm\u00b3 system, {model.binding_energy} binding energy, {model.number_of_nanoparticles} Nanoparticles,\n'
                f'{model.nanoparticle_radius} nm Nanoparticle Radius, {model.number_of_ligands} Ligands, Ligand length {model.ligand_length} nm,\n'
                f'{model.number_of_receptors} Receptors, {model.receptor_length} nm Receptor length, {model.binding_distance} Binding distance')
            model.run(steps=number_of_seconds)  # 3600 for 1 hour
            print(f'There were {model.count} reactions')
            print(f'The surface coverage is {model.surface_coverage}')
            variable_finals.append(model.surface_coverage)
            time_data.append(np.array(model.coverage))
        mean_time = np.mean(time_data, axis=0)
        error_time = np.std(time_data, axis=0)
        data[f'{model.binding_distance} nm binding distance'] = np.array(
            [list(range(0, model.time + 1)), mean_time, error_time])
        mean_coverage = np.mean(np.array(variable_finals))
        print(f'The mean surface coverage is {mean_coverage}')
        errors.append(np.std(np.array(variable_finals)))
        means.append(np.mean(np.array(variable_finals)))
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Surface Coverage')
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        plt.fill_between(value[0], value[1] - value[2], value[1] + value[2], alpha=0.2)
    plt.legend()
    plt.show()
    second_variable_plot('Binding distance (nm)', 'Surface Coverage', d, means, errors)

# def experiment_variable():
#     coverage = [0]
#     number_of_seconds = 100  # i.e. 1 hour = 3600 seconds
#     model = Master(dimension=1000, binding_energy=25, time_unit=10e-3, number_of_receptors=250, receptor_length=100,
#                        number_of_nanoparticles=50, number_of_ligands=100, nanoparticle_radius=50, ligand_length=7, binding_distance=4)
#     model.create_receptors()  # 100 nm for receptor
#     model.create_nanoparticles_and_ligands()  # 1-2 nm for ligand  # 95 particles
#     print(f'{model.dimension} nm\u00b3 system, {model.binding_energy} binding energy\n'
#           f'{model.number_of_nanoparticles} Nanoparticles, {model.nanoparticle_radius} nm Nanoparticle Radius, {model.number_of_ligands} Ligands,\n'
#           f'Ligand length {model.ligand_length} nm, {model.number_of_receptors} Receptors, {model.receptor_length} nm Receptor length')
#     model.run(steps=number_of_seconds)  # 3600 for 1 hour
#     print(f'There were {model.count} reactions')
#     print(f'The surface coverage is {model.surface_coverage}')
#     coverage.append(model.surface_coverage)
#     '''Surface coverage v Variable value'''
#     plt.title(f'{model.dimension} nm\u00b3 system, {model.binding_energy} binding energy\n'
#               f'{model.number_of_nanoparticles} Nanoparticles, {model.number_of_ligands} Ligands, {model.ligand_length} nm Ligand length \n'  # {model.number_of_receptors} Receptors,
#               f'{model.receptor_length} nm Receptor length')
#     plt.xlabel('Number of Receptors')
#     plt.ylabel('End Surface Coverage')
#     c.insert(0, 0)
#     plt.plot(c, coverage)
#     plt.show()


# experiment_variable()


def second_variable_plot(x, y, list1, list2, errors):
    plt.xlabel(x)
    plt.ylabel(y)
    list1.insert(0, 0)
    list2.insert(0, 0)
    errors.insert(0, 0)
    plt.errorbar(list1, list2, yerr=errors)
    plt.show()

if __name__=="__main__":
    binding_energy()