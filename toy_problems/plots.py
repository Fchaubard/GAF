import jax
import jax.numpy as jnp
import plotly.graph_objects as go

def plot_true_function(x: jax.Array, y: jax.Array, fn, n_contours=10, colorscale='Viridis'):
    # Create a meshgrid
    X, Y = jnp.meshgrid(x, y)

    Z = fn(jnp.stack([X, Y], axis=-1))

    fig = go.Figure(data=[
        go.Contour(
            z=Z,
            x=x,
            y=y,
            ncontours=n_contours,
            colorscale=colorscale,
            contours_coloring='lines',
            showscale=False,
            line_width=1,
            contours={
                "showlabels":True,
                "labelfont":{
                    "size":12,
                    "color":'black'
                }
            }
        )
    ])
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
    ))

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_xaxes(range=[x.min(), x.max()], title_text='X')
    fig.update_yaxes(range=[y.min(), y.max()], title_text='Y')

    fig.update_layout(width=500, height=500)

    return fig

def add_optimization_path(fig, path, color='red'):
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', line=dict(color=color, width=2), showlegend=False))