import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filepath_348 = "ex3/ana_348K"
filepath_298 = "ex3/ana_298K"

def plot_all(filepath, name):
    solutemp = pd.read_csv(filepath + "/ene_ana/solutemp2.dat", skiprows=1, header=None,sep="\s+", engine="python", dtype=np.float64)
    print(solutemp.head())
    solvtemp = pd.read_csv(filepath + "/ene_ana/solvtemp2.dat", skiprows=1, header=None,sep="\s+", engine="python", dtype=np.float64)
    print(solvtemp.head())
    totene = pd.read_csv(filepath + "/ene_ana/totene.dat", skiprows=1, header=None,sep="\s+", engine="python", dtype=np.float64)
    print(totene.head())
    totkin = pd.read_csv(filepath + "/ene_ana/totkin.dat", skiprows=1, header=None,sep="\s+", engine="python", dtype=np.float64)
    print(totkin.head())
    totpot = pd.read_csv(filepath + "/ene_ana/totpot.dat", skiprows=1, header=None,sep="\s+", engine="python", dtype=np.float64)
    print(totpot.head())
    pressu = pd.read_csv(filepath + "/ene_ana/pressu.dat", skiprows=1, header=None,sep="\s+", engine="python", dtype=np.float64)
    print(pressu.head())
    rmsd = pd.read_csv(filepath + "/rmsd/rmsd.dat", skiprows=9, header=None,sep="\s+", engine="python", dtype=np.float64)
    print(rmsd.head())
    rmsf = pd.read_csv(filepath + "/rmsf/rmsf.dat", usecols=lambda column: column != 2, skiprows=7, header=None,sep="\s+", engine="python", dtype=np.float64)
    print(rmsf.head())

    #plot all thermodynamics
    if (name == "348 K"):
        plot_single(solutemp, "Simulation Time in [ps]", "Solute Temperature in [K]", "Solute Temperature at " + name, name+"_solutemp.png", ylim=(315, 390), color="royalblue")
        plot_single(solvtemp, "Simulation Time in [ps]", "Solvent Temperature in [K]", "Solvent Temperature at " + name, name+"_solvtemp.png", ylim=(315, 390), color="royalblue")
    else:
        plot_single(solutemp, "Simulation Time in [ps]", "Solute Temperature in [K]", "Solute Temperature at " + name, name+"_solutemp.png", ylim=(265, 340))
        plot_single(solvtemp, "Simulation Time in [ps]", "Solvent Temperature in [K]", "Solvent Temperature at " + name, name+"_solvtemp.png", ylim=(265, 340))

    plot_single(totene, "Simulation Time in [ps]", "Total Energy in [kJ/mol]", "Total Energy at " + name, name+"_totene.png", color="royalblue")
    plot_single(totpot, "Simulation Time in [ps]", "Total Potential Energy in [kJ/mol]", "Total Potential Energy at " + name, name+"_totpot.png")
    plot_single(pressu, "Simulation Time in [ps]", "Pressure in [atm]", "Pressure at "+name, name+"_pressu.png")
    plot_single(rmsd, "Simulation Time in [ps]", "RMSD in [nm]", "RMSD at "+name, name+"_rmsd.png", ylim=(0.08, 0.22), color="royalblue")
    plot_single(rmsf, "Residue Number", "RMSF in [nm]", "RMSF at "+name, name+"_rmsf.png")
    

    return (solutemp, solvtemp, totene, totkin, totpot, pressu, rmsd, rmsf)


def plot_single(data, x_label, y_label, title, name, color="blue", xlim=None, ylim=None):
    data = data.to_numpy()
    fig, ax = plt.subplots()
    ax.plot(data[:,0], data[:,1], color=color)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xlabel(x_label, labelpad=20, fontsize=12)
    plt.ylabel(y_label, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(name)

def plot_double(data1, data2, x_label, y_label, title, name, color1="blue", color2="red", label1=None, label2=None, xlim=None, ylim=None, scatter = False):
    data1 = data1.to_numpy()
    data2 = data2.to_numpy()
    fig, ax = plt.subplots()
    if (scatter == False):
        ax.plot(data1[:,0], data1[:,1], '--bo', color=color1, label=label1)
        ax.plot(data2[:, 0], data2[:,1], '--bo',color=color2, label=label2)
    else:
        ax.scatter(data1[:,0], data1[:,1], color=color1, label=label1)
        ax.scatter(data2[:, 0], data2[:,1], color=color2, label=label2)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xlabel(x_label, labelpad=20, fontsize=12)
    plt.ylabel(y_label, labelpad=20, fontsize=12)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(name)


evolution_data = pd.read_csv("../evolution.csv", skiprows=0, header=None,sep=",", engine="python", dtype=np.float64)
plot_single(evolution_data, "Timestep", "L_Inf", "Hmm", "evolution")


 
'''filepath_tser = "ex2/ana/tser/"
filepath_ene = "ex2/ana/ene_ana/"

sim = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

tser_data = pd.DataFrame(columns=sim)
ele_solusolu_data = pd.DataFrame(columns=sim)
ele_solusolv_data = pd.DataFrame(columns=sim)

#get colormaps
colors = cm.plasma(np.linspace(0, 0.9, 9))

print("\n\nReading Data for end-to-end distance")
for sim_run in sim:
    print("Reading Data from run: " + sim_run)
    sim_run_filepath = filepath_tser + "analysis_results_" + sim_run + ".out"
    sim_run_data = pd.read_csv(
        sim_run_filepath,
        skiprows=2,
        skipfooter=2,
        header=None,
        sep="\s+",
        engine="python",
        dtype=np.float64,
    )
    tser_data[sim_run] = sim_run_data.to_numpy()[:, 1]
    tser_data["Time"] = sim_run_data.to_numpy()[:, 0]
print(tser_data.head())
print(tser_data.tail())

print("\n\nReading Data for solusolu analysis")
for sim_run in sim:
    print("Reading Data from run: " + sim_run)
    sim_run_filepath = filepath_ene + "ele_solusolu" + sim_run + ".dat"
    sim_run_data = pd.read_csv(
        sim_run_filepath,
        skiprows=2,
        skipfooter=2,
        header=None,
        sep="\s+",
        engine="python",
        dtype=np.float64,
    )
    ele_solusolu_data[sim_run] = sim_run_data.to_numpy()[:, 1]
    ele_solusolu_data["Time"] = sim_run_data.to_numpy()[:, 0]
print(ele_solusolu_data.head())
print(ele_solusolu_data.tail())


print("\n\nReading Data for solusolu analysis")
for sim_run in sim:
    print("Reading Data from run: " + sim_run)
    sim_run_filepath = filepath_ene + "ele_solusolv" + sim_run + ".dat"
    sim_run_data = pd.read_csv(
        sim_run_filepath,
        skiprows=2,
        skipfooter=2,
        header=None,
        sep="\s+",
        engine="python",
        dtype=np.float64,
    )
    ele_solusolv_data[sim_run] = sim_run_data.to_numpy()[:, 1]
    ele_solusolv_data["Time"] = sim_run_data.to_numpy()[:, 0]
print(ele_solusolv_data.head())
print(ele_solusolv_data.tail())

print("\n\nAll data read, starting plotting...")

#plotting end-to-end
data = tser_data.to_numpy()
fig, axs = plt.subplots(5, 2, figsize=(8, 10))

counter = 0
ylim = (0.25, 2.35)
for ax in axs.flat:
    if (counter != 9):
        ax.plot(data[:,9], data[:,counter], color=colors[counter])
        ax.grid(True)
        ax.set_title(sim[counter], fontsize=13)
        ax.set_ylim(ylim)
        #ax.set(xlabel='Simulation Time in [ps]', ylabel='End-to-end distance in [nm]')
        counter += 1
    else:
        ax.axis('off')
#add x and y labels
fig.text(0.5, 0.03, 'Simulation Time in [ps]', ha='center',fontsize = 15)
fig.text(0.03, 0.5, 'End-to-end Distance in [nm]', va='center', rotation= 'vertical', fontsize=15)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Leave space for labels

plt.savefig("images/tser.png")

#plotting solusolu 
data =ele_solusolu_data.to_numpy()
fig, axs = plt.subplots(5, 2, figsize=(8, 10))

counter = 0
ylim = (-1250, -425)
for ax in axs.flat:
    if (counter != 9):
        ax.plot(data[:,9], data[:,counter], color=colors[counter])
        ax.grid(True)
        ax.set_title(sim[counter], fontsize=13)
        ax.set_ylim(ylim)
    else:
        ax.axis('off')
    #ax.set(xlabel='Simulation Time in [ps]', ylabel='End-to-end distance in [nm]')
    counter += 1
#add x and y labels
fig.text(0.5, 0.03, 'Simulation Time in [ps]', ha='center',fontsize = 15)
fig.text(0.03, 0.5, 'Solute-Solute Potential Energy in [kJ/mol]', va='center', rotation= 'vertical', fontsize=15)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Leave space for labels

plt.savefig("images/solusolu.png")

#plotting solusolv
data =ele_solusolv_data.to_numpy()
fig, axs = plt.subplots(5, 2, figsize=(8, 10))

counter = 0
ylim = (0.25, 2.35)
for ax in axs.flat:
    if (counter != 9):
        ax.plot(data[:,9], data[:,counter], color=colors[counter])
        ax.grid(True)
        ax.set_title(sim[counter], fontsize=13)
        #ax.set_ylim(ylim)
    else:
        ax.axis('off')
    #yax.set(xlabel='Simulation Time in [ps]', ylabel='End-to-end distance in [nm]')
    counter += 1
#add x and y labels
fig.text(0.5, 0.03, 'Simulation Time in [ps]', ha='center',fontsize = 15)
fig.text(0.03, 0.5, 'Solute-Solvent Potential Energy in [kJ/mol]', va='center', rotation= 'vertical', fontsize=15)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Leave space for labels

plt.savefig("images/solusolv.png")

fig,ax = plt.subplots()
ax.plot(data[:25000,9], data[:25000,6], color='blue')
ax.grid(True)
ax.set_title("Solute-Solvent Interaction, Simulation G, First 25 Trajectories")
ax.set(xlabel='Simulation Time in [ps]', ylabel='Solute-Solvent Electrostatic Interaction in [kJ/mol]')
plt.tight_layout()
plt.savefig('images/unconverged')'''


