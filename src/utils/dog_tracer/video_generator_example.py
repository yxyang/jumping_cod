import dash
from dash import html, dcc, Output, Input, callback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import io
from datetime import datetime
from base64 import b64encode

# Create a Dash application
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Input(id='frame-count', type='number', value=10, min=1, max=10000, step=1),
    html.Button('Generate Video', id='generate-button'),
    dcc.Interval(id='progress-interval', interval=500, n_intervals=0),  # Periodic interval
    html.Div(id='progress-bar', children="Progress: 0%"),  # Progress bar placeholder
    html.Div(id='video-container')
])

# Store for the progress
progress = {'current': 0, 'total': 1}  # Global dictionary to track progress

# Generate frames based on the input data
def generate_frames(frame_count):
    progress['total'] = frame_count
    for i in range(frame_count):
        fig, ax = plt.subplots()
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x + np.pi * i / frame_count)
        ax.plot(x, y)
        ax.set_title(f"Frame {i + 1}")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        yield np.array(Image.open(buf))
        progress['current'] = i + 1

# Callback to update the progress bar
@callback(
    Output('progress-bar', 'children'),
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_progress(n):
    if progress['total'] == 0:  # Avoid division by zero
        return "Progress: 0%"
    progress_percentage = (progress['current'] / progress['total']) * 100
    return f"Progress: {progress_percentage:.0f}%"

# Callback to generate video
@callback(
    Output('video-container', 'children'),
    Input('generate-button', 'n_clicks'),
    [dash.dependencies.State('frame-count', 'value')],
    prevent_initial_call=True
)
def update_output(n_clicks, frame_count):
    if n_clicks is None:
        return dash.no_update, True

    # Reset progress
    progress['current'] = 0
    progress['total'] = frame_count

    # Generating video from frames
    output = io.BytesIO()
    with imageio.get_writer(output, format='mp4', fps=2) as writer:
        for frame in generate_frames(frame_count):
            writer.append_data(frame)

    output.seek(0)
    encoded_video = b64encode(output.read()).decode('ascii')
    timestamp = datetime.now().timestamp()  # Unique timestamp

    # Create a video tag to display the generated video, with a dynamic ID
    video_container = html.Div([
        html.Video(
            controls=True,
            children=[
                html.Source(src=f'data:video/mp4;base64,{encoded_video}', type='video/mp4')
            ],
            style={'width': '100%'}
        )
    ], id=f"video-container-{timestamp}")  # Dynamic ID for the container

    # Progress completed, disable interval
    return video_container

if __name__ == '__main__':
    app.run_server(debug=True)
