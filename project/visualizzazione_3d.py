import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import pyvista as pv
import os


# ===============================
# PREPROCESSING
# ===============================

def preprocess(data, use_log=False, use_percentile=False):

    # Rimuove dimensioni 1 → (1,3,128,128,128) → (128,128,128)
    data = np.squeeze(data)

    if data.ndim == 4:
        data = data[0]
    if data.ndim != 3:
        raise ValueError("Il file non contiene un cubo 3D dopo squeeze().")

    if use_log:
        data = np.log10(data + 1e-8)

    if use_percentile:
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, 99)
    else:
        vmin = data.min()
        vmax = data.max()

    return data, vmin, vmax


# ===============================
# 1️⃣ Animazione slice
# ===============================

def animate_slices(cube, vmin, vmax, base_name, output_dir="visualizzazione_patches"):

    fig, ax = plt.subplots()
    img = ax.imshow(cube[0, :, :],
                    origin='lower',
                    cmap='hot',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(img)
    ax.set_title("Slice 0")

    def update(frame):
        img.set_data(cube[frame, :, :])
        ax.set_title(f"Slice {frame}")
        return img,

    ani = FuncAnimation(fig,
                        update,
                        frames=cube.shape[0],
                        interval=80)
    

    # 2. Crea la cartella se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Costruisci il percorso completo del file
    out_name = f"{base_name}_animazione.gif"
    save_path = os.path.join(output_dir, out_name)

    
    print(f"Salvataggio in corso: {out_name}...")
    # Nota: richiede ffmpeg installato
    ani.save(save_path, writer='pillow', fps=15)
    print("Salvataggio completato.")

    plt.show()


# ===============================
# 2️⃣ Griglia slice
# ===============================

def static_grid(cube, vmin, vmax, base_name, output_dir="visualizzazione_patches"):

    indices = np.linspace(0, cube.shape[0]-1, 3, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for col, idx in enumerate(indices):

        # Asse 0 (XY)
        axes[0, col].imshow(cube[idx, :, :],
                            origin='lower',
                            cmap='hot',
                            vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"Piano RA Dec- slice {idx}")
        axes[0, col].axis("off")

        # Asse 1 (XZ)
        axes[1, col].imshow(cube[:, idx, :],
                            origin='lower',
                            cmap='hot',
                            vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f"Piano RA Freq- slice {idx}")
        axes[1, col].axis("off")

        # Asse 2 (YZ)
        axes[2, col].imshow(cube[:, :, idx],
                            origin='lower',
                            cmap='hot',
                            vmin=vmin, vmax=vmax)
        axes[2, col].set_title(f"Piano Dec Freq- slice {idx}")
        axes[2, col].axis("off")

   

    # 2. Crea la cartella se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Costruisci il percorso completo del file
    out_name = f"{base_name}_griglia_statico.png"
    save_path = os.path.join(output_dir, out_name)

    plt.tight_layout()

    # 4. Salva usando il percorso completo
    plt.savefig(save_path, dpi=300)

    print(f"Immagine salvata in: '{save_path}'")
    plt.show()



# ===============================
# 3️⃣ Slider interattivo
# ===============================

def interactive_slider(cube, vmin, vmax, base_name, output_dir="visualizzazione_patches"):

    # Assicurati che la directory di output esista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img = ax.imshow(cube[0, :, :],
                    origin='lower',
                    cmap='hot',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(img)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider,
                    'Slice',
                    0,
                    cube.shape[0]-1,
                    valinit=0,
                    valstep=1)

    def update(val):
        idx = int(slider.val)
        img.set_data(cube[idx, :, :])
        ax.set_title(f"Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    save_path = f"{output_dir}/{base_name}_slider_interattivo.png"
    plt.savefig(save_path, dpi=300)
    print(f"Immagine salvata in: '{save_path}'")


# ===============================
# 4️⃣ Volume Rendering 3D
# ===============================

def volume_rendering(cube, base_name, output_dir="visualizzazione_patches", use_log=False):
    # 1. Directory setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Pre-processing Astronomico (Log Scale)
    # Fondamentale per far emergere le strutture 3D deboli nel rendering
    cube_processed = np.squeeze(cube)
    if use_log:
        # Trasliamo per avere il minimo a 0 ed evitare log(0)
        eps = 1e-8
        cube_processed = np.log10(cube_processed - cube_processed.min() + eps)

    # 3. Normalizzazione basata sui Percentili
    # Questo serve a definire quali valori mappare sulla colormap 'hot'
    v_min = np.percentile(cube_processed, 5)   # Tagliamo un po' di rumore di fondo
    v_max = np.percentile(cube_processed, 99.5) # Evitiamo che picchi isolati oscurino tutto

    # 4. Setup della Grid PyVista
    grid = pv.ImageData()
    grid.dimensions = cube_processed.shape
    grid.spacing = (1, 1, 1)
    
    # Clippiamo i dati tra v_min e v_max per un rendering pulito
    cube_clipped = np.clip(cube_processed, v_min, v_max)
    grid.point_data["values"] = cube_clipped.flatten(order="F")

    # 5. Rendering
    plotter = pv.Plotter(off_screen=True) # Imposta True se lavori su Leonardo senza display
    
    # 'opacity' è fondamentale nel volume rendering: 
    # 'linear' o 'sigmoid' aiutano a vedere "dentro" il cubo
    plotter.add_volume(
        grid, 
        cmap="hot", 
        clim=[v_min, v_max], # Forza la scala colori sui nostri percentili
        opacity="sigmoid",   # Rende i valori bassi più trasparenti di quelli alti
        shade=True
    )
    
    filename = f"{base_name}_volume_rendering.png"
    save_path = os.path.join(output_dir, filename)
    
    # Sostituisci la sezione 6 con questa:
    plotter.show(screenshot=save_path, auto_close=False)
    plotter.close()
    
    print(f"Volume Rendering salvato in: '{save_path}' (Log: {use_log})")



# ===============================
# 5️⃣ Isosuperficie
# ===============================

def isosurface(cube, base_name, output_dir="visualizzazione_patches"):

    # Assicurati che la directory di output esista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    grid = pv.ImageData()
    grid.dimensions = cube.shape
    grid.spacing = (1, 1, 1)
    grid.point_data["values"] = cube.flatten(order="F")

    threshold = np.percentile(cube, 99)
    contours = grid.contour([threshold])
    
    plotter = pv.Plotter()

    if contours.n_points == 0:
        print(f"DEBUG: Mesh vuota per {base_name}. Soglia {threshold:.2f} troppo alta o dati piatti.")
        # Opzionale: aggiungi un testo o un cubo vuoto invece di crashare
        plotter.add_text("Sorgente non rilevata con soglia 99.5%", font_size=10, color="red")
        plotter.add_mesh(grid.outline(), color="gray", opacity=0.3)
    else:
        plotter.add_mesh(contours, color="cyan", opacity=0.8, label=f"Soglia: {threshold:.2f}")
        plotter.add_mesh(grid.outline(), color="white", opacity=0.2)

    
    
    filename = f"{base_name}_isosurface.png"
    save_path = f"{output_dir}/{filename}"
    print(f"Salvataggio screenshot in corso...")
    plotter.show(auto_close=False) # Apre la finestra e aspetta che tu la chiuda con 'q'
    plotter.screenshot(save_path)   # Scatta la foto all'ultima posizione della camera
    plotter.close()
    print(f"Screenshot salvato come {save_path}")


# ===============================
# MAIN
# ===============================

def main(path, choice = "4"):
    
    if not os.path.exists(path):
        print("Errore: File non trovato.")
        return

    # ESTRAZIONE NOME FILE
    # Esempio: "data/esperimento_01.npy" -> "esperimento_01"
    base_name = os.path.splitext(os.path.basename(path))[0]
    
    data = np.load(path)

    print("Shape originale:", data.shape)

    
    cube, vmin, vmax = preprocess(data,
                                  use_log=False,
                                  use_percentile=True)
    print("Shape finale:", cube.shape)

    print("\nScegli visualizzazione:")
    print("1 - Animazione slice")
    print("2 - Griglia slice statiche")
    print("3 - Slider interattivo")
    print("4 - Volume rendering 3D")
    print("5 - Isosuperficie 3D")

    

    if choice == "1":
        animate_slices(cube, vmin, vmax, base_name,    output_dir="visualizzazione_patches")

    elif choice == "2":
        static_grid(cube, vmin, vmax, base_name, output_dir="visualizzazione_patches")

    elif choice == "3":
        interactive_slider(cube, vmin, vmax, base_name, output_dir="visualizzazione_patches")

    elif choice == "4":
        volume_rendering(cube, base_name, output_dir="visualizzazione_patches")

    elif choice == "5":
        isosurface(cube, base_name, output_dir="visualizzazione_patches")

    else:
        print("Scelta non valida.")


if __name__ == "__main__":
    path = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/inputs/16x128x128_stride128_cont_dev/train/npy_patches/patch_000025.npy"
    main(path, choice="1")
