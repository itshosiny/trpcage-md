import os
from pymol import cmd

RUNS = {"T300": "runs/T300", "T360": "runs/T360"}
COLORS = {"T300": "marine", "T360": "firebrick"}
OUT = "images"
WIDTH, HEIGHT, DPI = 1600, 1200, 300


def style_scene():
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("cartoon_transparency", 0)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_side_chain_helper", 1)
    cmd.set("stick_radius", 0.2)
    cmd.set("ray_trace_mode", 1)          # cartoon with black outline
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 1)
    cmd.set("depth_cue", 0)


def show_protein(obj, color=None):
    cmd.hide("everything", obj)
    cmd.show("cartoon", obj)
    if color:
        cmd.color(color, obj)
    else:
        cmd.spectrum("resi", "rainbow", obj)
    # Trp6 (core of the cage) and the prolines that pack against it
    cmd.show("sticks", f"{obj} and (resn TRP or resi 12+17+18+19) and not name N+C+O")
    cmd.color("orange", f"{obj} and resn TRP and elem C")
    cmd.hide("everything", f"{obj} and hydro")


def render(name):
    path = os.path.join(OUT, name)
    cmd.ray(WIDTH, HEIGHT)
    cmd.png(path, dpi=DPI)
    print(f"saved {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    cmd.reinitialize()
    style_scene()

    # Native reference = minimised starting structure of the 300 K run
    cmd.load(os.path.join(RUNS["T300"], "protein_start.pdb"), "native")

    finals = []
    for name, folder in RUNS.items():
        cmd.load(os.path.join(folder, "protein_start.pdb"), name)
        cmd.load_traj(os.path.join(folder, "protein.xtc"), name, state=1)
        n = cmd.count_states(name)
        print(f"{name}: {n} frames")
        # Copy the last frame into its own object and fit it onto the native structure
        final = f"{name}_final"
        cmd.create(final, name, n, 1)
        cmd.align(f"{final} and name CA", "native and name CA", cycles=0)
        finals.append(final)
        cmd.disable(name)                  # keep the full trajectory loaded but hidden

    # 1. Native structure
    cmd.disable("all")
    cmd.enable("native")
    show_protein("native")
    cmd.orient("native")
    view = cmd.get_view()                  # same camera for every image
    render("start.png")

    # 2. Final frame at each temperature
    for final in finals:
        cmd.disable("all")
        cmd.enable(final)
        show_protein(final)
        cmd.set_view(view)
        render(f"{final}.png")

    # 3. Overlay: native (grey) + final frames (blue = 300 K, red = 360 K)
    cmd.disable("all")
    cmd.enable("native")
    show_protein("native", "grey80")
    cmd.set("cartoon_transparency", 0.5, "native")
    for final in finals:
        cmd.enable(final)
        show_protein(final, COLORS[final.split("_")[0]])
    cmd.set_view(view)
    render("overlay_final.png")

    # Leave the session ready for interactive viewing: play the 360 K movie
    cmd.disable("all")
    cmd.enable("T360")
    show_protein("T360")
    cmd.intra_fit("T360 and name CA", 1)
    cmd.set_view(view)
    cmd.mplay()
    print("Done. Images are in the 'images' folder. Type 'mstop' to stop the movie.")


main()
