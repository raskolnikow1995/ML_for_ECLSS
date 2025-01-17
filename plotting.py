import seaborn as sns
import matplotlib.pyplot as plt

# Plotting function that does not create new figures, using Seaborn
def plot_simulations(df, y, xlim, time_unit="h"):
    """Plots all simulations in gray using Seaborn, without showing the legend initially."""
    
    fig, ax = plt.subplots()
    
    df = scale_time(df, time_unit)
    
    # Loop through simulations and plot using Seaborn lineplot
    for Simulation in df["Simulation"].unique():
        subset = df[df["Simulation"] == Simulation]
        sns.lineplot(
            data=subset,
            x="Time",
            y=y,
            label=Simulation,
            color="lightgray",
            alpha=1,
            linewidth=1,
            ax=ax
        )
    
    ax.set_xlabel(f"Time [{time_unit}]")
    ax.set_ylabel(y)
    ax.set_title(y)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    return fig, ax

def select_sim(sim, ax):
    palette = sns.color_palette()

    # Remove all lines from the legend initially
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[], labels=[])

    # Set all lines to lightgray and default linewidth
    for line in ax.lines:
        line.set_color("lightgray")
        line.set_linewidth(1)  # Set linewidth for all lines to 1 initially
        line.set_zorder(1)  # Set the z-order for all lines to the background
    
    # Find the line corresponding to the selected simulation
    for line in ax.lines:
        if line.get_label() == sim:
            # Change color and linewidth of the selected line
            line.set_color(palette[0])  # Highlighted color
            line.set_linewidth(1)  # Thicker line for selected simulation
            line.set_zorder(2)  # Bring the selected line to the front

    # Update legend for the selected simulation
    ax.legend([line for line in ax.lines if line.get_label() == sim],
              [sim], loc="upper left")
    
def scale_time(df, time_unit="s"):
    """
    Scales the 'Time' column of the DataFrame to the specified unit: 's', 'h', or 'd'.
    """
    # Define scaling factors relative to seconds
    scaling_factors = {"s": 1, "h": 3600, "d": 86400}
    
    if time_unit not in scaling_factors:
        raise ValueError("Invalid time unit. Choose 's' (seconds), 'h' (hours), or 'd' (days).")
    
    # Normalize time back to seconds before applying scaling
    if "OriginalTime" not in df.columns:
        df["OriginalTime"] = df["Time"]  # Save original time for normalization
    
    df["Time"] = df["OriginalTime"] / scaling_factors[time_unit]
    return df