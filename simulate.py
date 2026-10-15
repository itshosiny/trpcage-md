import argparse
import os
import sys
import time

import mdtraj
from mdtraj.reporters import XTCReporter
import openmm as mm
import openmm.app as app
import openmm.unit as u


def pick_platform(name):
    """Use the requested platform, or the fastest one available."""
    if name != "auto":
        return mm.Platform.getPlatformByName(name)
    available = [mm.Platform.getPlatform(i).getName()
                 for i in range(mm.Platform.getNumPlatforms())]
    for p in ("CUDA", "OpenCL", "CPU"):
        if p in available:
            return mm.Platform.getPlatformByName(p)
    return mm.Platform.getPlatformByName("Reference")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", required=True, help="input structure (e.g. 1L2Y.pdb)")
    ap.add_argument("--temp", type=float, default=300.0, help="temperature, K")
    ap.add_argument("--ns", type=float, default=20.0, help="production length, ns")
    ap.add_argument("--equil_ps", type=float, default=200.0, help="restrained equilibration, ps")
    ap.add_argument("--save_ps", type=float, default=10.0, help="trajectory save interval, ps")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--platform", default="auto", help="auto | CUDA | OpenCL | CPU")
    ap.add_argument("--seed", type=int, default=0, help="velocity seed (0 = random)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    T = args.temp * u.kelvin
    dt = 4.0 * u.femtoseconds          # allowed by hydrogen mass repartitioning
    steps_per_ps = int(round(1.0 * u.picosecond / dt))

    # ---- 1. Build the system ------------------------------------------------
    pdb = app.PDBFile(args.pdb)        # multi-model NMR files: first model is used
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    # Remove existing hydrogens and rebuild them consistently at pH 7.
    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None
                     and a.element.symbol == "H"])
    modeller.addHydrogens(ff, pH=7.0)
    n_protein_atoms = modeller.topology.getNumAtoms()
    modeller.addSolvent(ff, model="tip3p", padding=1.0 * u.nanometer,
                        ionicStrength=0.15 * u.molar, neutralize=True)
    n_atoms = modeller.topology.getNumAtoms()
    print(f"System: {n_protein_atoms} protein atoms, {n_atoms} atoms total")

    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * u.nanometer,
                             constraints=app.HBonds,
                             hydrogenMass=1.5 * u.amu)
    system.addForce(mm.MonteCarloBarostat(1.0 * u.bar, T))

    # Harmonic restraints on protein heavy atoms, switched off via global 'k'.
    restraint = mm.CustomExternalForce("0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2")
    restraint.addGlobalParameter("k", 1000.0)          # kJ/mol/nm^2
    for p in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)
    protein_idx = list(range(n_protein_atoms))
    atoms = list(modeller.topology.atoms())
    for i in protein_idx:
        if atoms[i].element.symbol != "H":
            restraint.addParticle(i, modeller.positions[i].value_in_unit(u.nanometer))
    system.addForce(restraint)

    integrator = mm.LangevinMiddleIntegrator(T, 1.0 / u.picosecond, dt)
    if args.seed:
        integrator.setRandomNumberSeed(args.seed)
    platform = pick_platform(args.platform)
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    print(f"Platform: {platform.getName()}")

    # ---- 2. Minimise and equilibrate -----------------------------------------
    print("Minimising...")
    sim.minimizeEnergy(maxIterations=5000)
    sim.context.setVelocitiesToTemperature(T, args.seed or 1)

    # Save the solvated and the protein-only starting structures.
    state = sim.context.getState(getPositions=True)
    with open(os.path.join(args.out, "start_solvated.pdb"), "w") as f:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
    full = mdtraj.load(os.path.join(args.out, "start_solvated.pdb"))
    full.atom_slice(protein_idx).save_pdb(os.path.join(args.out, "protein_start.pdb"))

    equil_steps = int(args.equil_ps * steps_per_ps)
    print(f"Restrained equilibration: {args.equil_ps:.0f} ps")
    sim.step(equil_steps)
    sim.context.setParameter("k", 0.0)                 # release restraints

    # ---- 3. Production ---------------------------------------------------------
    prod_steps = int(args.ns * 1000 * steps_per_ps)
    save_every = max(1, int(args.save_ps * steps_per_ps))
    sim.reporters.append(XTCReporter(os.path.join(args.out, "protein.xtc"),
                                     save_every, atomSubset=protein_idx))
    sim.reporters.append(app.StateDataReporter(
        os.path.join(args.out, "log.csv"), save_every, time=True,
        potentialEnergy=True, temperature=True, density=True, speed=True))
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, max(save_every, prod_steps // 20), step=True, time=True,
        temperature=True, speed=True, remainingTime=True, totalSteps=equil_steps + prod_steps))
    sim.reporters.append(app.CheckpointReporter(
        os.path.join(args.out, "checkpoint.chk"), 100 * save_every))

    print(f"Production: {args.ns} ns at {args.temp:.0f} K ({prod_steps} steps)")
    t0 = time.time()
    sim.step(prod_steps)
    hours = (time.time() - t0) / 3600
    print(f"Done in {hours:.2f} h -> {args.out}")


if __name__ == "__main__":
    main()
