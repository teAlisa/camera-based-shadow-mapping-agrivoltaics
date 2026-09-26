"""
visualization.py

Purpose:
    Helper functions for saving and plotting shadow-analysis results.

Main uses:
    - Save shadow probability maps as colored PNG images.
    - Save raw shadow probability arrays as .npy files.
    - Show quick visual checks for one processed image.
    - Convert ground-grid results into 2D maps.
    - Plot ground-grid heatmaps with optional solar panel row overlays.

Notes:
    PNG probability maps are mainly for visual inspection.
    The .npy probability arrays preserve the actual values and should be used
    for downstream analysis.

    Ground-grid plots can be shown either in grid-cell coordinates
    or in real-world meter coordinates.
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_probability_map(
    shadow_prob: np.ndarray,
    save_path: str,
) -> None:
    """
    Saves a shadow probability array as a colored PNG image.

    Parameters
    ----------
    shadow_prob : np.ndarray
        2D shadow probability map with values from 0 to 1.
        Pixels outside the valid mask may be NaN.
    save_path : str
        Output path for the colored PNG image.

    Notes
    -----
    NaN values are treated as 0 for visualization only.

    This PNG should not be used for numerical analysis because it is converted
    to 8-bit color. Use save_probability_array() for preserving raw values.
    """

    # Replace NaN with 0 only for visualization.
    prob = np.nan_to_num(shadow_prob, nan=0)

    # Convert probability values from 0-1 to 0-255 for OpenCV colormap.
    prob_8bit = (prob * 255).clip(0, 255).astype(np.uint8)

    prob_color = cv2.applyColorMap(
        prob_8bit,
        cv2.COLORMAP_JET,
    )

    cv2.imwrite(save_path, prob_color)


def save_probability_array(
    shadow_prob: np.ndarray,
    save_path: str,
) -> None:
    """
    Saves raw shadow probability values as a NumPy array.

    Parameters
    ----------
    shadow_prob : np.ndarray
        2D shadow probability map with values from 0 to 1 and possible NaNs.
    save_path : str
        Output path for the .npy file.

    Notes
    -----
    This preserves actual probability values and NaNs. These .npy files are
    used later for polygon-level and ground-grid analysis.
    """

    np.save(save_path, shadow_prob.astype(np.float32))


def show_shadow_result(
    image_bgr: np.ndarray,
    final_mask: np.ndarray,
    shadow_prob: np.ndarray,
    pred_shadow: np.ndarray,
    title: str = "Shadow result",
) -> None:
    """
    Shows a quick visual check for one processed image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Original image in OpenCV BGR format.
    final_mask : np.ndarray
        Binary valid ground mask.
    shadow_prob : np.ndarray
        Shadow probability map.
    pred_shadow : np.ndarray
        Binary predicted shadow mask.
    title : str
        Figure title.

    Notes
    -----
    This function is mainly for debugging and visual inspection in notebooks
    or interactive runs.
    """

    fig, ax = plt.subplots(1, 4, figsize=(22, 6))

    ax[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original image")
    ax[0].axis("off")

    ax[1].imshow(final_mask, cmap="gray")
    ax[1].set_title("Ground mask")
    ax[1].axis("off")

    im = ax[2].imshow(
        shadow_prob,
        cmap="viridis_r",
        vmin=0,
        vmax=1,
    )
    ax[2].set_title("Shadow probability")
    ax[2].axis("off")
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    ax[3].imshow(pred_shadow, cmap="gray")
    ax[3].set_title("Binary shadow mask")
    ax[3].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def make_grid_map(
    df,
    value_col: str,
) -> np.ndarray:
    """
    Converts a long-format grid DataFrame into a 2D array.

    Parameters
    ----------
    df : pd.DataFrame
        Grid table containing cell_x, cell_y, and the selected value column.
    value_col : str
        Column to place into the 2D grid map.

    Returns
    -------
    np.ndarray
        2D array where rows correspond to cell_y and columns correspond
        to cell_x.

    Expected columns
    ----------------
    cell_x : int
        Grid column index.
    cell_y : int
        Grid row index.
    value_col : float
        Value to plot for each cell.

    Notes
    -----
    This format is useful for plotting heatmaps of grid-level results.
    """

    n_cols = int(df["cell_x"].max()) + 1
    n_rows = int(df["cell_y"].max()) + 1

    grid_map = np.full((n_rows, n_cols), np.nan)

    for _, row in df.iterrows():
        x = int(row["cell_x"])
        y = int(row["cell_y"])
        grid_map[y, x] = row[value_col]

    return grid_map


def _save_figure_if_requested(save_path: str | None) -> None:
    """
    Saves the current matplotlib figure if save_path is provided.

    Parameters
    ----------
    save_path : str or None
        Output file path. If None, the figure is not saved.
    """

    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_ground_map(
    grid_map: np.ndarray,
    title: str,
    colorbar_label: str,
    save_path: str | None = None,
    vmin: float = 0,
    vmax: float = 1,
    figsize: tuple = (6, 10),
    cmap_name: str = "viridis_r",
) -> None:
    """
    Plots and optionally saves a ground-grid heatmap.

    Parameters
    ----------
    grid_map : np.ndarray
        2D grid map to plot.
    title : str
        Plot title.
    colorbar_label : str
        Label for the colorbar.
    save_path : str or None
        Output path for the figure. If None, the figure is not saved.
    vmin : float
        Minimum color scale value.
    vmax : float
        Maximum color scale value.
    figsize : tuple
        Figure size.
    cmap_name : str
        Matplotlib colormap name.

    Notes
    -----
    For shade maps, viridis_r makes higher shadow values appear darker.
    NaN cells are shown in light gray.
    """

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="lightgray")

    masked_map = np.ma.masked_invalid(grid_map)

    plt.figure(figsize=figsize)

    plt.imshow(
        masked_map,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel("cell_x")
    plt.ylabel("cell_y")
    plt.tight_layout()

    _save_figure_if_requested(save_path)

    plt.close()


def save_grid_map_from_dataframe(
    df,
    value_col: str,
    title: str,
    colorbar_label: str,
    save_path: str,
    vmin: float = 0,
    vmax: float = 1,
    figsize: tuple = (6, 10),
    cmap_name: str = "viridis_r",
) -> np.ndarray:
    """
    Converts a DataFrame to a 2D grid map and saves it as a PNG figure.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format grid DataFrame.
    value_col : str
        Column to plot.
    title : str
        Plot title.
    colorbar_label : str
        Label for the colorbar.
    save_path : str
        Output path for the saved figure.
    vmin : float
        Minimum color scale value.
    vmax : float
        Maximum color scale value.
    figsize : tuple
        Figure size.
    cmap_name : str
        Matplotlib colormap name.

    Returns
    -------
    np.ndarray
        The 2D grid map used for plotting.
    """

    grid_map = make_grid_map(df, value_col)

    plot_ground_map(
        grid_map=grid_map,
        title=title,
        colorbar_label=colorbar_label,
        save_path=save_path,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        cmap_name=cmap_name,
    )

    return grid_map


def plot_map_with_panel_rows(
    df,
    value_col: str,
    panel_rows_grid,
    title: str | None = None,
    colorbar_label: str | None = None,
    vmin: float = 0,
    vmax: float = 1,
    panel_color: str = "black",
    save_path: str | None = None,
    cmap_name: str = "viridis_r",
    figsize: tuple = (6, 10),
) -> None:
    """
    Plots a grid heatmap with solar panel row lines.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format grid DataFrame.
    value_col : str
        Column to plot as the heatmap value.
    panel_rows_grid : pd.DataFrame
        Table with panel row start/end coordinates in grid-cell coordinates.
    title : str or None
        Plot title. If None, value_col is used.
    colorbar_label : str or None
        Colorbar label. If None, value_col is used.
    vmin : float
        Minimum color scale value.
    vmax : float
        Maximum color scale value.
    panel_color : str
        Color for panel row lines and labels.
    save_path : str or None
        Output path for the saved figure.
    cmap_name : str
        Matplotlib colormap name.
    figsize : tuple
        Figure size.

    Notes
    -----
    This version plots panel rows using grid-cell coordinates:
        start_cell_x, start_cell_y, end_cell_x, end_cell_y

    For shade maps, viridis_r makes higher shadow values appear darker.
    """

    grid_map = make_grid_map(df, value_col)

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="lightgray")

    plt.figure(figsize=figsize)

    plt.imshow(
        np.ma.masked_invalid(grid_map),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if colorbar_label is None:
        colorbar_label = value_col

    plt.colorbar(label=colorbar_label)

    # Draw each panel row as a line between projected start and end cells.
    for _, row in panel_rows_grid.iterrows():
        plt.plot(
            [row["start_cell_x"], row["end_cell_x"]],
            [row["start_cell_y"], row["end_cell_y"]],
            color=panel_color,
            linewidth=3,
        )

        plt.scatter(
            [row["start_cell_x"], row["end_cell_x"]],
            [row["start_cell_y"], row["end_cell_y"]],
            color=panel_color,
            s=45,
        )

        plt.text(
            row["start_cell_x"] + 0.2,
            row["start_cell_y"] + 0.2,
            f'Row {int(row["row_id"])}',
            color=panel_color,
            fontsize=10,
        )

    if title is None:
        title = value_col

    plt.title(title)
    plt.xlabel("cell_x")
    plt.ylabel("cell_y")
    plt.tight_layout()

    _save_figure_if_requested(save_path)

    plt.close()


def plot_map_with_panel_rows_meters(
    df,
    value_col: str,
    panel_rows_grid,
    title: str | None = None,
    colorbar_label: str | None = None,
    vmin: float = 0,
    vmax: float = 1,
    panel_color: str = "black",
    save_path: str | None = None,
    cmap_name: str = "viridis_r",
    figsize: tuple = (7, 10),
) -> None:
    """
    Plots a ground-grid heatmap in real-world meter coordinates with panel rows.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format grid DataFrame with x_center_cm and y_center_cm columns.
    value_col : str
        Column to plot as the heatmap value.
    panel_rows_grid : pd.DataFrame
        Table with panel row start/end coordinates in world centimeters.
    title : str or None
        Plot title. If None, value_col is used.
    colorbar_label : str or None
        Colorbar label. If None, value_col is used.
    vmin : float
        Minimum color scale value.
    vmax : float
        Maximum color scale value.
    panel_color : str
        Color for panel row lines and labels.
    save_path : str or None
        Output path for the saved figure.
    cmap_name : str
        Matplotlib colormap name.
    figsize : tuple
        Figure size.

    Notes
    -----
    This version plots the map using real-world coordinates in meters.
    It is easier to interpret physically than the cell_x/cell_y version.
    """

    grid_map = make_grid_map(df, value_col)

    # Convert cell center coordinates from centimeters to meters.
    x_centers_m = np.sort(df["x_center_cm"].unique()) / 100
    y_centers_m = np.sort(df["y_center_cm"].unique()) / 100

    cell_size_x = np.median(np.diff(x_centers_m))
    cell_size_y = np.median(np.diff(y_centers_m))

    # Extent tells imshow how to label the image axes in real-world units.
    extent = [
        min(x_centers_m) - cell_size_x / 2,
        max(x_centers_m) + cell_size_x / 2,
        min(y_centers_m) - cell_size_y / 2,
        max(y_centers_m) + cell_size_y / 2,
    ]

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="lightgray")

    plt.figure(figsize=figsize)

    plt.imshow(
        np.ma.masked_invalid(grid_map),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
    )

    if colorbar_label is None:
        colorbar_label = value_col

    plt.colorbar(label=colorbar_label)

    # Draw panel rows using world coordinates converted from centimeters
    # to meters.
    for _, row in panel_rows_grid.iterrows():
        plt.plot(
            [row["start_world_x_cm"] / 100, row["end_world_x_cm"] / 100],
            [row["start_world_y_cm"] / 100, row["end_world_y_cm"] / 100],
            color=panel_color,
            linewidth=3,
        )

        plt.scatter(
            [row["start_world_x_cm"] / 100, row["end_world_x_cm"] / 100],
            [row["start_world_y_cm"] / 100, row["end_world_y_cm"] / 100],
            color=panel_color,
            s=45,
        )

        plt.text(
            row["start_world_x_cm"] / 100 + 0.1,
            row["start_world_y_cm"] / 100 + 0.1,
            f'Row {int(row["row_id"])}',
            color=panel_color,
            fontsize=10,
        )

    if title is None:
        title = value_col

    plt.title(title)
    plt.xlabel("X position, m")
    plt.ylabel("Y position, m")
    plt.tight_layout()

    _save_figure_if_requested(save_path)

    plt.close()