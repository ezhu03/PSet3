import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# Define your sub-region of interest
row_start, row_end = 10, 40
col_start, col_end = 10, 40

# 1. Define paths
base_dir = "ldos"
input_dir = os.path.join(base_dir)  # your text files are here
output_dir_a = os.path.join(base_dir, "ldos_heatmap")
output_dir_b = os.path.join(base_dir, "ldos_height")
fft_output_dir = os.path.join(base_dir, "fft_plots")
# 2. Create the output directory if it does not exist
os.makedirs(output_dir_a, exist_ok=True)
os.makedirs(output_dir_b, exist_ok=True)
os.makedirs(fft_output_dir, exist_ok=True)
# 3. Get all .txt files in base_dir
files = [f for f in os.listdir(base_dir) if f.endswith(".txt")]

# 4. Sort the filenames by the numeric value that appears before .txt
#    (Assuming the entire filename is '0.txt', '1.txt', ... '10.txt', etc.)
files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
subregion_averages = []
subregion_data = []
file_indices = []
# 3. Loop over each text file in the directory
for filename in files:
    # We only want to process .txt files (or however your files are named)
    if filename.endswith(".txt"):
        file_index = int(os.path.splitext(filename)[0].split('_')[-1])
        file_path = os.path.join(input_dir, filename)
        print(file_path)
        # 4. Load the data into a 2D NumPy array
        #    Depending on your data format, adjust delimiter or skip headers if necessary
        data = np.loadtxt(file_path, delimiter=",")
        
        # 5. Plot the 2D heatmap
        plt.figure(figsize=(6, 5))
        plt.imshow(data, cmap="viridis", origin="lower", aspect="auto",vmin=0,vmax=0.005)
        
        # 6. Add a color legend (colorbar) for intensity
        cbar = plt.colorbar()
        cbar.set_label("Local density of states (a.u.)")  # Change label as appropriate
        
        # 7. Add a title or label (e.g., file index)
        #    You might parse an index out of the filename if needed
        plt.title(f"LDOS Heatmap: {filename}")
        plt.xlabel("X index")
        plt.ylabel("Y index")
        
        # 8. Save the figure in the output directory
        #    You can save as PNG, PDF, etc.
        outname = os.path.splitext(filename)[0] + "_heatmap.png"
        plt.savefig(os.path.join(output_dir_a, outname), dpi=150, bbox_inches="tight")
        
        # 9. Close the figure to free memory
        plt.close()
        # 5. Prepare a meshgrid for the surface plot axes
        #    data.shape gives (rows, cols). 
        #    Typically the shape might be something like (Ny, Nx).
        Ny, Nx = data.shape
        x = np.arange(Nx)
        y = np.arange(Ny)
        X, Y = np.meshgrid(x, y)
        
        # 6. Create a figure with 3D axes
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 7. Plot the surface
        #    `rstride` and `cstride` control row/column “strides” for drawing
        surf = ax.plot_surface(X, Y, data, color='blue', edgecolor='none',alpha=0.5)
        
        # 9. Title and axis labels
        ax.set_title(f"LDOS Surface Plot: {filename}")
        ax.set_xlabel("X Index")
        ax.set_ylabel("Y Index")
        ax.set_zlabel("LDOS")
        
        # 10. Save the figure
        output_filepath = os.path.join(output_dir_b, f"{os.path.splitext(filename)[0]}_surface.png")
        plt.savefig(output_filepath, dpi=150, bbox_inches="tight")
        
        # 11. Close the figure to free memory
        plt.close()

        # 6. Extract the sub-region
        subregion = data[row_start:row_end, col_start:col_end]
        subregion_data.append(subregion)
    
        # 7. Compute average
        avg_val = np.mean(subregion)
    
        # 8. Store results
        subregion_averages.append(avg_val)
        file_indices.append(file_index)
        # 1) Compute 2D FFT of the subregion
        fft2d = np.fft.fft2(subregion)
    
        # 2) Shift the zero frequency component to the center
        fft2d_shifted = np.fft.fftshift(fft2d)
    
        # 3) Compute the power spectrum (magnitude squared)
        power_spectrum = np.abs(fft2d_shifted)**2
    
        # ========== PLOT THE POWER SPECTRUM ==========
        plt.figure(figsize=(6, 5))
        plt.imshow(power_spectrum, cmap='viridis', origin='lower',vmin=0,vmax=0.005)
        plt.colorbar(label="Power (|FFT|^2)")
        plt.title(f"2D FFT Power Spectrum: {filename} (subregion)")
        plt.xlabel("kx (pixel index)")
        plt.ylabel("ky (pixel index)")
        plt.tight_layout()
    
    # ========== SAVE THE PLOT ==========
    # Extract the numeric index to use in the output file name
        outname = f"fft_subregion_{file_index}.png"
        outpath = os.path.join(fft_output_dir, outname)
        plt.savefig(outpath, dpi=150)
    
    # Close the figure to free memory
        plt.close()
# 9. Plot the average LDOS vs file index
plt.figure(figsize=(6, 4))
plt.plot(file_indices, subregion_averages, marker='o', linestyle='-')
plt.xlabel("File Index")
plt.ylabel("Average LDOS in Sub-region")
plt.title("Average Sub-region LDOS vs. File Index")
plt.grid(True)
plt.tight_layout()

# 10. Save or show the plot
plt.savefig("average_subregion_ldos.png", dpi=150)
plt.show()

Ny, Nx = subregion_data[0].shape  # shape of the subregion
x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(x, y)

# We will store a reference to the "surface" object so we can update it
surf = None

def init():
    """
    Initialize the plot with the first frame (frame=0).
    """
    ax.clear()
    ax.set_xlabel("Subregion X Index")
    ax.set_ylabel("Subregion Y Index")
    ax.set_zlabel("LDOS")
    ax.set_title("LDOS Subregion Animation")

    # Plot first subregion
    Z = subregion_data[0]
    # plot_surface returns a Poly3DCollection
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none')
    
    # Set a nice viewing angle (optional)
    ax.view_init(elev=30, azim=-60)
    
    return (surf,)

def update(frame):
    """
    Update the surface plot for the given frame index.
    """
    # Clear the old plot so we can draw a fresh surface
    ax.clear()
    
    # Axes labels/title
    ax.set_xlabel("Subregion X Index")
    ax.set_ylabel("Subregion Y Index")
    ax.set_zlabel("LDOS")
    ax.set_title(f"LDOS Subregion Animation - Frame {frame}")

    # Plot the new subregion data
    Z = subregion_data[frame]
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none')
    
    # Adjust the view angle if you like
    ax.view_init(elev=30, azim=-60)
    
    return (surf,)

# Create the animation using FuncAnimation
ani = animation.FuncAnimation(
    fig,           # Figure to animate
    update,        # Update function
    frames=len(subregion_data),  # Number of frames
    init_func=init, 
    blit=False,    # For 3D surface, blit=False is typically safer
    interval=500   # Time in ms between frames
)

# ========== SAVE OR SHOW THE ANIMATION ==========

# 1) To display inline in a Jupyter notebook, just call:
# plt.show()

# 2) To save as an MP4 (requires FFmpeg installed):
ani.save("ldos_subregion_animation.mp4", writer="ffmpeg", dpi=150)

plt.show()