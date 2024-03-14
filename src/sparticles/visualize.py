import plotly.graph_objs as go
from typing import List, Tuple, Union, Optional
import os
import plotly.express as px
import math
import pandas as pd
from plotly.subplots import make_subplots


############################################################################################################

# types of particles in the dataset
PARTICLE_TYPES = ['jet1', 'jet2', 'jet3', 'b1', 'b2', 'lepton', 'energy']

# markers to use in the scatter plots
PARTICLE_MARKERS = dict(zip(PARTICLE_TYPES, ['circle', 'circle', 'circle', 'circle', 'circle', 'circle', 'square']))

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

    Args:
        g (torch_geometric.data.Data): The particle event to plot.
        size_is_pt (bool, optional): Whether to use the particle's Pt as the marker size. Defaults to True.
        show_energy (bool, optional): Whether to include the energy particle in the plot. Defaults to True.
        show_edges (bool, optional): Whether to draw edges between the particles. Defaults to False.
        edges_weight (Optional[List], optional): A matrix (numpy, torch, python) representing the edge weights. More specifically, 
                    matrix[i][j] should be the weight of the edge (i,j). Defaults to None.
        save_to (Union[os.PathLike, str, bytes], optional): Path to save the plot to. If None, the plot is displayed. Defaults to None.
        **kwargs: Additional arguments to pass to the plotly figure.

    Returns:
        fig (plotly.graph_objs._figure.Figure): The plotly figure object.
    """
    
    if g.x.shape[1] != 6:
        raise ValueError(f"Expected g.x to have 6 columns, got {g.x.shape[1]} instead. Maybe you applied a transform that added columns to the data?")

    particles = PARTICLE_TYPES.copy()
    if g.x.shape[0] == 6:
        particles.remove('jet3')

    df = pd.DataFrame(g.x.numpy(), columns=['Pt', 'eta', 'phi', 'quantile', 'mass', 'metsig'])
    df['particle'] = particles
    df['marker'] = df['particle'].map(PARTICLE_MARKERS)

    # Replace nan values with 'NaN' to avoid errors in plotly
    df.fillna('NaN', inplace=True)

    # set energy coordinates to 0.
    df.iloc[-1,[1,2]] = 0

    # remove energy particle if not requested
    if not show_energy:
        df = df[df['particle'] != 'energy']

    # create figure
    fig = px.scatter(
            df,
            y='eta', 
            x='phi',
            symbol=df['marker'],
            color='particle',
            color_discrete_map=PARTICLE_COLORS,
            size='Pt' if size_is_pt else [DEFAULT_MARKER_SIZE]*df.shape[0],
            hover_name='particle',
            hover_data=['quantile','mass'],
            # text='particle' if display_particle_name else None,
            labels={'color': "particle", 'size': "Pt"}
            )

    fig.update_traces(
        marker_line_color='black')
        #marker_line_width=2)
    
    # add edges
    # TODO this can be made more efficient but it was the easiest way to do it now
    if show_edges:
        # connect each particle to all the others with a line
        for i in range(df.shape[0]):
            for j in range(i+1, df.shape[0]):
                edge_weight = edges_weights[i][j] if edges_weights is not None else None
                edge_weight = float(edge_weight)
                fig.add_trace(
                    go.Scatter(
                        x=[df.iloc[i,2], df.iloc[j,2]], 
                        y=[df.iloc[i,1], df.iloc[j,1]], 
                        mode='lines', 
                        line=dict(color='black', width=edge_weight or 1), 
                        showlegend=False,
                        hovertext=str(edge_weight) if edge_weight else 'none',
                        text=str(edge_weight) if edge_weight else 'none',
                        hoverinfo='text',
                        ))
                
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
    

    # save image 
    # if save_to is a directory, we save the image there with the event id as name, else we save it to the path
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


def display_attention_matrix(graph, attention_scores, save_to: Union[os.PathLike, str, bytes] = None, **kwargs):
    """
    Displays the attention scores of a graph.

    Args:
        graph (torch_geometric.data.Data): The graph to display the attention scores of.
        attention_scores (torch.Tensor): The attention scores to display.
        save_to (Union[os.PathLike, str, bytes], optional): The path to save the image to. If None, the image is displayed instead of saved. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the plotly `update_layout` function.

    Returns:
        fig (plotly.graph_objs.Figure): The plotly figure object.
    """
    fig = go.Figure(data=go.Heatmap(z=attention_scores, x=graph.event_id, y=graph.event_id))
    fig.update_layout(title="Attention matrix", xaxis_title="Event id", yaxis_title="Event id", **kwargs)
    
    if save_to is not None:
        is_html = save_to.endswith('.html')
        if is_html:
            fig.write_html(save_to)
        else:
            fig.write_image(save_to)
    else: 
        fig.show()
    return fig

# WARNING this is not used so not tested
def plot_event_3d(g, xyz: Tuple = (1, 2, 0), aspectmode:str = 'cube', save_to: Union[os.PathLike, str, bytes] = None,  **kwargs):
    """
    Plots a 3D visualization of a particle event.

    Parameters:
    -----------
    g : `torch_geometric.data.Data`
        The particle event to visualize.
    xyz : Tuple[int, int, int], optional
        The indices of the columns in `g.x` to use as x, y, and z coordinates, respectively.
        Default is (0, 1, 2).
    aspectmode : str, optional
        The aspect mode of the 3D plot. Default is 'cube'.
    save_to : Union[os.PathLike, str, bytes], optional 
        The path to save the image to. If None, the image is displayed instead of saved.
        Default is None.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the `go.Layout` constructor.

    Returns:
    --------
    None
    """

    x_col, y_col, z_col = xyz[0], xyz[1], xyz[2]

    # Select node and edge coordinates
    Xn = g.x[:, x_col].tolist() # x-coordinates of nodes
    Yn = g.x[:, y_col].tolist() # y-coordinates of nodes
    Zn = g.x[:, z_col].tolist() # z-coordinates of nodes

    Xe = g.x[:, x_col][g.edge_index.t().unique()].tolist()  # x-coordinates of edge ends
    Ye = g.x[:, y_col][g.edge_index.t().unique()].tolist()  # y-coordinates of edge ends
    Ze = g.x[:, z_col][g.edge_index.t().unique()].tolist()  # z-coordinates of edge ends

    # TODO we could use other attributes for size or color
    # colors = g.x[:, 3].tolist() # color of nodes
    # size = g.x[:, 3].tolist() # size of nodes

    trace1=go.Scatter3d(x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        line=dict(color='rgb(125,125,125)', width=1),
                        hoverinfo='none'
                        )
    
    trace2=go.Scatter3d(x=Xn,
                        y=Yn,
                        z=Zn,
                        mode='markers',
                        # name='actors',
                        marker=dict(symbol='circle', # [‘circle’, ‘circle-open’, ‘cross’, ‘diamond’, ‘diamond-open’, ‘square’, ‘square-open’, ‘x’]
                                    # size=8,
                                    # color=colors,
                                    colorscale='Viridis',
                                    line=dict(color='rgb(50,50,50)', width=0.5)
                                    ),
                        # text=labels,
                        showlegend=True,
                        hoverinfo='text',
                        hovertext=list(zip(Xn, Yn, Zn)) # TODO add other node features
                        )


    # https://plotly.com/python/reference/layout/scene/
    background_distance = 5 # a margin that keeps the background far from points. TODO this should be a parameter
    xaxis=dict(
        showbackground=True,
        range= [min(Xn)-background_distance, max(Xn)+background_distance],
        autorange=False,
        showline=True,
        zeroline=True,
        showgrid=True,
        showticklabels=True,
        title='x',)
    yaxis=dict(
        showbackground=True,
        range=[min(Yn)-background_distance, max(Yn)+background_distance],
        autorange=False,
        showline=True,
        zeroline=True,
        showgrid=True,
        showticklabels=True,
        title='y',)
    zaxis=dict(
        showbackground=True,
        # range= [min(Zn)-background_distance, max(Zn)+background_distance],
        showline=True,
        zeroline=True,
        showgrid=True,
        showticklabels=True,
        title='z',)
        

    layout = go.Layout(
         title=f"Event {g.event_id}",
         showlegend=False,
         scene=dict(
             xaxis=xaxis,
             yaxis=yaxis,
             zaxis=zaxis,
            aspectmode=aspectmode
            ),
        **kwargs,) # https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.html#plotly.graph_objects.layout.Margin
    
    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)
    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    
    # save or display image
    if save_to is not None:
        is_html = save_to.endswith('.html')
        if is_html:
            fig.write_html(save_to)
        else:
            fig.write_image(save_to)
    else: 
        fig.show()  