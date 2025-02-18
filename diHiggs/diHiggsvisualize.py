import plotly.graph_objs as go
from typing import List, Tuple, Union, Optional
import os
import plotly.express as px
import math
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np


############################################################################################################

# types of particles in the dataset
PARTICLE_TYPES = ['tau', 'lepton', 'b1', 'b2', 'energy', 'jet']

# markers to use in the scatter plots
PARTICLE_MARKERS = dict(zip(PARTICLE_TYPES, ['circle', 'circle', 'circle', 'circle', 'square', 'circle']))

# default size of the markers in scatter plots
DEFAULT_MARKER_SIZE = 3

# range of the axes in the scatter plots
XRANGE = [-math.pi - 1, math.pi + 1] 
YRANGE = [-7, +7]

# dictionary <particle, color> of colors of the particles in the scatter plots
PARTICLE_COLORS = dict(zip(PARTICLE_TYPES, px.colors.qualitative.Plotly))

# number of columns to use in the multiplot
MULTIPLOT_NUM_COLS = 3


############################################################################################################

def _plot_event_2d(g, 
                   size_is_pt :bool = True, 
                   show_energy: bool = True, 
                   show_edges: bool = False,
                   edges_weights: Optional[List] = None,
                   save_to: Union[os.PathLike, str, bytes] = None, 
                   **kwargs):
    """
    Plots a 2D scatter plot of the particles in an event.
    """
    particles = PARTICLE_TYPES.copy()
    if g.x.shape[0] == 5:
        particles.remove('jet')

    df = pd.DataFrame(g.x.numpy(), columns=['Pt', 'eta', 'phi'])
    df['particle'] = particles
    df['marker'] = df['particle'].map(PARTICLE_MARKERS)

    # Replace nan values with 'NaN' to avoid errors in plotly
    df.fillna('NaN', inplace=True)

    # set energy coordinates to 0.
    df.iloc[4, [1, 2]] = 0

    # Ensure the energy particle is included with visible size
    if show_energy:
        energy_row = df[df['particle'] == 'energy']
        if energy_row.shape[0] > 0:
            energy_pt = energy_row['Pt'].iloc[0]
            if energy_pt == 0:  # Ensure energy particle has a visible size
                df.loc[df['particle'] == 'energy', 'Pt'] = DEFAULT_MARKER_SIZE

    # Remove energy particle if not requested
    if not show_energy:
        df = df[df['particle'] != 'energy']

    # Create the figure
    fig = px.scatter(
            df,
            y='eta', 
            x='phi',
            symbol=df['marker'],
            color='particle',
            color_discrete_map=PARTICLE_COLORS,
            size='Pt' if size_is_pt else [DEFAULT_MARKER_SIZE]*df.shape[0],
            hover_name='particle',
            labels={'color': "particle", 'size': "Pt"}
            )

    fig.update_traces(
        marker_line_color='black')
    
    # Add edges (if requested)
    if show_edges:
     for i in range(df.shape[0]):
        for j in range(i+1, df.shape[0]):  # Ensures j > i
            if edges_weights is not None:
                edge_weight = float(edges_weights[i][j])
            else:
                edge_weight = 1.0  # Default edge width
            fig.add_trace(
                go.Scatter(
                    x=[df.iloc[i, 2], df.iloc[j, 2]], 
                    y=[df.iloc[i, 1], df.iloc[j, 1]], 
                    mode='lines', 
                    line=dict(color='black', width=edge_weight),
                    showlegend=False,
                    hovertext=str(edge_weight),
                    text=str(edge_weight),
                    hoverinfo='text',
                )
            )

    fig.update_layout(
        title=f"Event id {g.event_id} <br><sup>(Energy displayed at origin, but it does not have coordinates)</sup>", 
        showlegend=True,
        xaxis_title="φ",
        yaxis_title="η",
        xaxis_range=XRANGE,
        yaxis_range=YRANGE,
        scattermode="group",
        scattergap=0.8,
        yaxis_zeroline=False, 
        xaxis_zeroline=False,
        **kwargs,)

    if save_to is not None:
        if os.path.isdir(save_to):
            save_to = os.path.join(save_to, f"event_{g.event_id}.png")
        else:
            is_html = save_to.endswith('.html')
            if is_html:
                fig.write_html(save_to)
            else:
                fig.write_image(save_to)

    return fig



def plot_event_2d(g, size_is_pt :bool = True, show_energy: bool = True, show_edges:bool=False, edges_weights: Optional[List] = None, save_to: Union[os.PathLike, str, bytes] = None, **kwargs):
    """
    Plots a 2D event display for a given graph or list of graphs.

    Args:
        g ( torch_geometric.data.Data or List[torch_geometric.data.Data]): The graph or list of graphs to plot.
        display_particle_name (bool, optional): Whether to display the particle name on the plot. Defaults to False.
        size_is_pt (bool, optional): Whether the size of the particle markers is proportional to the particle transverse momentum. Defaults to True.
        show_energy (bool, optional): Whether to display the particle energy. Defaults to True.
        show_edges (bool, optional): Whether to display the edges between the particles. Defaults to False.
        edges_weights (Optional[List], optional): A matrix or list of matrices (numpy, torch, python) representing the edge weights. More specifically, matrix[i][j] should be the weight of the edge (i,j). Defaults to None.
        save_to (Union[os.PathLike, str, bytes], optional): The file path to save the plot to. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the plotly `update_layout` function.

    Returns:
        fig (plotly.graph_objs.Figure): The plotly figure object.
    """
    
    if isinstance(g, list):
        
        if isinstance(edges_weights, list) and len(edges_weights) != len(g):
            raise ValueError("If g is a list, edges_weights must be a list with the same length")
        
        plot_cols = MULTIPLOT_NUM_COLS if MULTIPLOT_NUM_COLS < len(g) else len(g)
        fig = make_subplots(rows=math.ceil(len(g)/plot_cols), cols=plot_cols, subplot_titles=[f"Event {g[i].event_id}" for i in range(len(g))])
        
        for graph_id, graph in enumerate(g):
            graph_plot = _plot_event_2d(graph, size_is_pt, show_energy, show_edges=show_edges, edges_weights=edges_weights[graph_id] if edges_weights is not None else None, save_to=None, **kwargs)
            for d in graph_plot['data']:
                d['showlegend'] = True if graph_id == 0 and d['mode'] != 'lines' else False
                fig.add_trace(d, row=math.ceil((graph_id+1)/plot_cols), col=(graph_id % plot_cols) + 1)
                

        fig.update_yaxes(range=YRANGE, zeroline=False, title_text="η")
        fig.update_xaxes(range=XRANGE, zeroline=False, title_text="φ")
        fig.update_layout(**kwargs)

        if save_to is not None:
            is_html = save_to.endswith('.html')
            if is_html:
                fig.write_html(save_to)
            else:
                fig.write_image(save_to)

        return fig
    else:
        return _plot_event_2d(g, size_is_pt, show_energy, show_edges, edges_weights, save_to, **kwargs)
