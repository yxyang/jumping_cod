# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import os
import pickle
from datetime import datetime
import time

import dash
from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output, State
from flask import Response
import imageio
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objs as go
import plotly.io as pio
import pybullet as p
from tqdm import tqdm

from isaacgym.torch_utils import to_torch
import torch

app = Dash(__name__)

list(pio.templates)  # Available templates
pio.templates.default = "seaborn"


def load_data(content):
  # logs = pickle.load(open(data_dir, 'rb'))
  content_type, content_string = content.split(',')
  decoded = base64.b64decode(content_string)
  logs = pickle.loads(decoded)
  for frame in logs:
    for key in frame:
      if type(frame[key]) == torch.Tensor and key != "num_clips":
        frame[key] = frame[key].cpu().numpy()
        if frame[key].shape[0] != 1:
          raise ValueError(
              f"Only 1 robot is supported at this time. Key is: {key}")
        frame[key] = frame[key][0]
  return logs


def create_upload_div():
  upload_div = html.Div(
      [
          html.H3("Upload Data"),
          dcc.Upload(
              id='upload_data',
              children=html.Div(['Drag and Drop or ',
                                 html.A('Select Files')]),
              style={
                  'width': '80%',
                  'height': '60px',
                  'lineHeight': '60px',
                  'borderWidth': '4px',
                  'borderStyle': 'dashed',
                  'borderRadius': '5px',
                  'textAlign': 'center',
                  'margin': '10px'
              },
              # Allow multiple files to be uploaded
              multiple=True),
          html.Button('Clear All Data',
                      id='clear_logs',
                      style={
                          'justifyContent': 'center',
                          'alignItems': 'center',
                          'margin': '10px'
                      })
      ],
      style={
          'width': '33%',
          'height': '15vw'
      },
  )
  selection_div = html.Div(
      [
          html.H3("Select Data to Plot"),
          dcc.Checklist(id="traj_name_selector", value=list(all_data.keys()))
      ],
      style={
          'width': '33%',
          'height': '15vw'
      },
  )

  video_div = html.Div(
      children=[
          html.H3("Video Trajectory", style={'height': '1vw'}),
          dcc.Dropdown(
              id='video_trajectory_dropdown',
              value=None,  # Default value
              options=[],
              style={'height': '2vw'}),
          dcc.Interval(id='progress-interval', interval=500,
                       n_intervals=0),  # Periodic interval
          html.Div(id='progress-bar',
                   children="Progress: 0%"),  # Progress bar placeholder
          html.Div(id='video-container')
      ],
      style={
          'width': '33%',
          'height': '15vw'
      },
  )
  return html.Div(
      [upload_div, selection_div, video_div],
      style={
          'width': '100vw',
          'display': 'flex',
          'height': '15vw'
      },
  )


@app.callback(Output('video_trajectory_dropdown', 'options'),
              Input('traj_name_selector', 'value'))
def video_trajectory_options(selected_trajectories):
  return [{'label': traj, 'value': traj} for traj in selected_trajectories]


def _foot_positions_in_hip_frame(motor_positions):
  motor_positions = motor_positions.reshape((4, 3))
  theta_ab = motor_positions[:, 0]
  theta_hip = motor_positions[:, 1]
  theta_knee = motor_positions[:, 2]
  l_up = 0.213
  l_low = 0.213
  l_hip = np.array([-1, 1, -1, 1]) * 0.08
  leg_distance = np.sqrt(l_up**2 + l_low**2 +
                         2 * l_up * l_low * np.cos(theta_knee))
  eff_swing = theta_hip + theta_knee / 2

  off_x_hip = -leg_distance * np.sin(eff_swing)
  off_z_hip = -leg_distance * np.cos(eff_swing)
  off_y_hip = l_hip

  off_x = off_x_hip
  off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
  off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
  return np.stack([off_x, off_y, off_z], axis=1)


hip_offset = np.array([[0.1881, -0.04675, 0.], [0.1881, 0.04675, 0.],
                       [-0.1881, -0.04675, 0.], [-0.1881, 0.04675, 0.]])


def get_foot_position(frame, body_frame=False):
  foot_position_body_frame = frame["foot_positions_in_base_frame"]
  if body_frame:
    return foot_position_body_frame

  base_rot_mat = p.getMatrixFromQuaternion(
      p.getQuaternionFromEuler((frame['base_orientation_rpy'][0],
                                frame['base_orientation_rpy'][1], 0)))
  base_rot_mat = np.array(base_rot_mat).reshape((3, 3))
  return base_rot_mat.dot(foot_position_body_frame.T).T


progress = {'current': 0, 'total': 1}  # Global dictionary to track progress


def generate_heightmap_video(data, prefix=""):
  # Create a video buffer in memory
  output = io.BytesIO()

  # Start the video writer
  writer = imageio.get_writer(output, format='mp4',
                              fps=20)  # Change FPS as needed
  # Generate frames and write directly to video
  progress['total'] = len(data)
  for idx, frame in enumerate(tqdm(data)):
    # Create a figure
    progress['current'] = idx + 1
    if "height_est" in frame:
      fig, ax = plt.subplots(figsize=(10, 5))
      plt.title(prefix + f"Timestamp: {frame['timestamp']:.2f}")
      x = np.linspace(-0.4, 0.8, 30)
      heights = frame["height_est"]
      plt.plot(x, heights, color="#8c564b")
      pitch = frame["base_orientation_rpy"][1]
      body_length = 0.1881
      hip_x, hip_z = body_length * np.cos(-pitch), body_length * np.sin(-pitch)

      foot_position = get_foot_position(frame, body_frame=False)
      front_x = (foot_position[0][0] + foot_position[1][0]) / 2
      front_z = (foot_position[0][2] + foot_position[1][2]) / 2
      rear_x = (foot_position[2][0] + foot_position[3][0]) / 2
      rear_z = (foot_position[2][2] + foot_position[3][2]) / 2
      plt.scatter([front_x, 0, rear_x], [front_z, 0, rear_z], color="#1f77b4")
      plt.plot([-hip_x, hip_x], [-hip_z, hip_z], color="#1f77b4")
      plt.plot([hip_x, front_x], [hip_z, front_z], '--', color="#1f77b4")
      plt.plot([-hip_x, rear_x], [-hip_z, rear_z], '--', color="#1f77b4")

      plt.ylim(-0.6, 0.4)
      plt.xlim(-0.5, 0.9)
      plt.gca().set_aspect('equal')

      # Save the plot to a BytesIO object
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      buf.seek(0)

      # Read image from buffer and add to video
      image = np.array(Image.open(buf))
      # print(image.shape)
      writer.append_data(image)

      # Close the figure to free memory
      plt.close(fig)
      buf.close()

  # Finish writing and close the video file
  writer.close()

  # Prepare the video for embedding in Colab notebook
  output.seek(0)
  mp4 = output.read()
  data_url = "data:video/mp4;base64," + base64.b64encode(mp4).decode()
  return data_url


@callback(Output('progress-bar', 'children'),
          Input('progress-interval', 'n_intervals'),
          prevent_initial_call=True)
def update_progress(n):
  if progress['total'] == 0:  # Avoid division by zero
    return "Progress: 0%"
  progress_percentage = (progress['current'] / progress['total']) * 100
  return f"Progress: {progress_percentage:.0f}%"


@app.callback(Output('video-container', 'children'),
              Input('video_trajectory_dropdown', 'value'),
              prevent_initial_call=True)
def render_videos(selected_traj):
  if selected_traj is None or not all_data:
    return dash.no_update, True

  print(f"To render trajectory: {selected_traj}")
  timestamp = datetime.now().timestamp()  # Unique timestamp

  video_content = generate_heightmap_video(all_data[selected_traj],
                                           prefix=selected_traj[-5:] + " ")

  return html.Div([
      html.Video(
          id='video-player',
          controls=True,
          children=[html.Source(src=video_content, type='video/mp4')],
          style={
              'width': 'auto',
              'maxHeight': '11vw'
          },
      ),
  ],
                  id=f"video-container-{timestamp}")


@app.callback(Output('traj_name_selector', 'options', allow_duplicate=True),
              [Input('clear_logs', 'n_clicks')],
              prevent_initial_call=True)
def clear_data(_):
  global all_data
  all_data = {}
  return []


@app.callback(Output('traj_name_selector', 'options', allow_duplicate=True),
              Input('upload_data', 'contents'),
              State('upload_data', 'filename'),
              prevent_initial_call=True)
def file_upload_callback(list_of_contents, list_of_names):
  if list_of_contents is None:
    return []
  global all_data
  for name, content in tqdm(zip(list_of_names, list_of_contents)):
    filename = os.path.splitext(name)[0]
    all_data[filename] = load_data(content)
  return list(all_data.keys())


def generate_timeseries_plot(traj_names: str,
                             attr_name="base_velocity",
                             dim=0,
                             title="Base Velocity: X",
                             xlabel="Time/s",
                             ylabel="m/s"):
  fig = go.Figure()
  for traj_name in traj_names:
    data = all_data[traj_name]
    ts = np.array([frame["timestamp"] for frame in data])
    y = np.array([frame[attr_name][dim] for frame in data])
    fig.add_scatter(x=ts, y=y, name=traj_name)
  fig.update_layout(
      title=title,
      xaxis_title=xlabel,
      yaxis_title=ylabel,
      title_x=0.5,  # This centers the title
      title_y=1.,  # This adjusts the vertical position of the title
      margin=dict(t=20, b=40, l=20, r=20),
      legend=dict(
          x=0,  # This sets the horizontal position of the legend
          y=1,  # This sets the vertical position of the legend
          bordercolor='rgba(255, 255, 255, 0.)',
          bgcolor='rgba(255, 255, 255, 0.5)',
          borderwidth=1),
      paper_bgcolor='rgba(255, 255, 255, 1.)',
  )
  return fig


def create_base_velocity_div():
  base_vel_div = html.Div(
      children=[
          html.H3('Base Velocity', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="base_vel_x", className="plot"),
                  dcc.Graph(id="base_vel_y", className='plot'),
                  dcc.Graph(id="base_vel_z", className='plot')
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return base_vel_div


@app.callback([
    Output('base_vel_x', 'figure'),
    Output('base_vel_y', 'figure'),
    Output('base_vel_z', 'figure')
], [Input('traj_name_selector', 'value')])
def update_base_velocity_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  vel_x_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="env_obs",
                                       dim=15,
                                       title="X")
  vel_y_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="env_obs",
                                       dim=16,
                                       title="Y")
  vel_z_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="env_obs",
                                       dim=17,
                                       title="Z")
  return vel_x_fig, vel_y_fig, vel_z_fig


@app.callback(Output('depth-trajectory-dropdown', 'options'),
              Input('traj_name_selector', 'value'))
def set_trajectory_options(selected_trajectories):
  return [{'label': traj, 'value': traj} for traj in selected_trajectories]


@app.callback(Output('timestamp-slider', 'max'),
              Output('timestamp-slider', 'marks'),
              Output('timestamp-slider', 'value'),
              Input('depth-trajectory-dropdown', 'value'),
              Input('timestamp-slider', 'value'))
def select_depth_traj(selected_trajectory, current_slider_value):
  if selected_trajectory is None:
    return 0, {}, 0
  global depth_images, depth_timestamps
  depth_timestamps = []
  depth_images = []
  for frame in all_data[selected_trajectory]:
    if "depth_image" in frame:
      depth_timestamps.append(frame['timestamp'])
      depth_images.append(frame['depth_image'])
  marks = {i: '' for i, timestamp in enumerate(depth_timestamps)}
  marks[0] = '0'
  marks[len(depth_timestamps) - 1] = f"{depth_timestamps[-1]:.2f}"

  if current_slider_value is not None:
    marks[
        current_slider_value] = f"{depth_timestamps[current_slider_value]:.2f}"

  return len(depth_timestamps) - 1, marks, current_slider_value or 0


@app.callback(Output('depth-image', 'figure'),
              [Input('timestamp-slider', 'value')])
def update_depth_image(selected_idx):
  if not depth_images:
    return dash.no_update
  image = depth_images[selected_idx]
  return {
      'data': [go.Heatmap(z=image[::-1], colorscale='gray', zmin=0, zmax=1)],
      'layout':
      go.Layout(title=f"Timestamp: {depth_timestamps[selected_idx]:.2f}",
                xaxis={'visible': False},
                yaxis={'visible': False},
                autosize=True,
                margin={
                    'l': 0,
                    'r': 0,
                    'b': 30,
                    't': 30
                },
                yaxis_constrain='domain')
  }


@app.callback(
    [Output('base_pos_xy', 'figure'),
     Output('base_pos_z', 'figure')], [Input('traj_name_selector', 'value')])
def update_base_position_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []

  scatters = []
  for traj_name in selected_traj_names:
    data = all_data[traj_name]
    base_pos = np.array([frame['base_position'] for frame in data])
    scatters.append(
        go.Scatter3d(x=base_pos[:, 0],
                     y=base_pos[:, 1],
                     z=base_pos[:, 2],
                     name=traj_name,
                     mode='lines'))

  layout = go.Layout(
      title_text="Space Trajectory",
      title_x=0.5,  # This centers the title
      title_y=1.,  # This adjusts the vertical position of the title
      margin=dict(t=20, b=40, l=20, r=20),
      legend=dict(
          x=0,  # This sets the horizontal position of the legend
          y=1,  # This sets the vertical position of the legend
          bgcolor='rgba(255, 255, 255, 0.5)',
          bordercolor='rgba(0, 0, 0, 0.1)',
          borderwidth=1),
      paper_bgcolor='rgba(255, 255, 255, 1.)',
      scene=dict(
          xaxis_title='x',
          yaxis_title='y',
          zaxis_title='z',
          aspectmode='manual',
          aspectratio=dict(x=1, y=1, z=1.),
      ))
  pos_fig = go.Figure(data=scatters, layout=layout)
  # pos_fig.update_layout()
  # pos_fig.update_yaxes(
  #     scaleanchor="x",
  #     scaleratio=1,
  # )
  # pos_fig.update_zaxes(
  #     scaleanchor="x",
  #     scaleratio=1,
  # )

  pos_z_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_position",
                                       dim=2,
                                       title="Z",
                                       ylabel="Height/m")
  return pos_fig, pos_z_fig


def create_base_position_div():
  base_pos_div = html.Div(
      children=[
          html.H3('Base Position', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="base_pos_xy",
                            className="plot",
                            style={'height': '20vh'}),
                  dcc.Graph(id="base_pos_z",
                            className="plot",
                            style={'height': '20vh'}),
                  html.Div(
                      children=[
                          dcc.Dropdown(
                              id='depth-trajectory-dropdown',
                              value=None,  # Default value
                              options=[]),
                          dcc.Graph(id='depth-image',
                                    className="plot",
                                    style={'height': '18vh'}),
                          dcc.Slider(id='timestamp-slider',
                                     min=0,
                                     max=1,
                                     value=0,
                                     marks={
                                         0: '0',
                                         1: '1'
                                     },
                                     step=1,
                                     className="plot",
                                     updatemode="drag")
                      ], )
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '30vh',
          # 'overflow': 'hidden'
      },
  )
  return base_pos_div


def create_base_orientation_div():
  base_vel_div = html.Div(
      children=[
          html.H3('Base Orientation', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="base_roll", className="plot"),
                  dcc.Graph(id="base_pitch", className='plot'),
                  dcc.Graph(id="base_yaw", className='plot')
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return base_vel_div


@app.callback([
    Output('base_roll', 'figure'),
    Output('base_pitch', 'figure'),
    Output('base_yaw', 'figure')
], [Input('traj_name_selector', 'value')])
def update_base_orientation_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  vel_x_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_orientation_rpy",
                                       dim=0,
                                       title="Roll",
                                       ylabel="rad")
  vel_y_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_orientation_rpy",
                                       dim=1,
                                       title="Pitch",
                                       ylabel="rad")
  vel_z_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_orientation_rpy",
                                       dim=2,
                                       title="Yaw",
                                       ylabel="rad")
  return vel_x_fig, vel_y_fig, vel_z_fig


def create_base_angvel_div():
  base_vel_div = html.Div(
      children=[
          html.H3('Base Angular Velocity', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="base_angvel_x", className="plot"),
                  dcc.Graph(id="base_angvel_y", className='plot'),
                  dcc.Graph(id="base_angvel_z", className='plot')
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return base_vel_div


@app.callback([
    Output('base_angvel_x', 'figure'),
    Output('base_angvel_y', 'figure'),
    Output('base_angvel_z', 'figure')
], [Input('traj_name_selector', 'value')])
def update_base_angvel_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  vel_x_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_angular_velocity",
                                       dim=0,
                                       title="X",
                                       ylabel="rad/s")
  vel_y_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_angular_velocity",
                                       dim=1,
                                       title="Y",
                                       ylabel="rad/s")
  vel_z_fig = generate_timeseries_plot(selected_traj_names,
                                       attr_name="base_angular_velocity",
                                       dim=2,
                                       title="Z",
                                       ylabel="rad/s")
  return vel_x_fig, vel_y_fig, vel_z_fig


def create_foot_contact_div():
  foot_contact_div = html.Div(
      children=[
          html.H3('Foot Contact', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="foot_contact", className="plot"),
                  dcc.Graph(id="foot_contact_force", className='plot'),
              ],
              className='container',
          )
      ],
      style={
          'width': '100vw',
          'height': '20vh',
          # 'overflow': 'hidden'
      },
  )
  return foot_contact_div


@app.callback([
    Output('foot_contact', 'figure'),
], [Input('traj_name_selector', 'value')])
def update_base_foot_contact_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []

  active_template = pio.templates[pio.templates.default]
  # Get the color sequence from the active template
  color_sequence = active_template.layout.colorway

  plots = []
  for idx, traj_name in enumerate(selected_traj_names):
    data = all_data[traj_name]
    ts = np.array([frame['timestamp'] for frame in data])
    foot_contact = np.array([frame['foot_contact_state'] for frame in data])
    t1 = go.Scatter(
        x=ts,
        y=foot_contact[:, 0] * 0.8 + 3,
        mode='lines',
        name=traj_name,
        legendgroup=traj_name,
        # legendgrouptitle_text=traj_name,
        line=dict(color=color_sequence[idx]))
    t2 = go.Scatter(x=ts,
                    y=foot_contact[:, 1] * 0.8 + 2,
                    mode='lines',
                    name="FL",
                    legendgroup=traj_name,
                    line=dict(color=color_sequence[idx]),
                    showlegend=False)
    t3 = go.Scatter(x=ts,
                    y=foot_contact[:, 2] * 0.8 + 1,
                    mode='lines',
                    name="RR",
                    legendgroup=traj_name,
                    line=dict(color=color_sequence[idx]),
                    showlegend=False)
    t4 = go.Scatter(x=ts,
                    y=foot_contact[:, 3] * 0.8 + 0,
                    mode='lines',
                    name="RL",
                    legendgroup=traj_name,
                    line=dict(color=color_sequence[idx]),
                    showlegend=False)
    plots.extend([t1, t2, t3, t4])

  foot_contact_fig = go.Figure(data=plots)
  foot_contact_fig.update_layout(
      title="Foot Contact",
      xaxis_title="time/s",
      yaxis_title="",
      title_x=0.5,  # This centers the title
      title_y=1.,  # This adjusts the vertical position of the title
      margin=dict(t=20, b=40, l=20, r=20),
      legend=dict(
          x=0,  # This sets the horizontal position of the legend
          y=1,  # This sets the vertical position of the legend
          bgcolor='rgba(255, 255, 255, 0.5)',
          bordercolor='rgba(0, 0, 0, 0.1)',
          borderwidth=1),
      paper_bgcolor='rgba(255, 255, 255, 1.)',
  )
  return [foot_contact_fig]


@app.callback([
    Output('foot_contact_force', 'figure'),
], [Input('traj_name_selector', 'value')])
def update_base_foot_contact_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []

  active_template = pio.templates[pio.templates.default]
  # Get the color sequence from the active template
  color_sequence = active_template.layout.colorway

  plots = []
  for idx, traj_name in enumerate(selected_traj_names):
    data = all_data[traj_name]
    ts = np.array([frame['timestamp'] for frame in data])
    foot_contact_force = np.array(
        [frame['foot_contact_force'] for frame in data])
    foot_contact_force = foot_contact_force / np.mean(foot_contact_force,
                                                      axis=0) / 5
    t1 = go.Scatter(
        x=ts,
        y=foot_contact_force[:, 0] * 0.8 + 3,
        mode='lines',
        name=traj_name,
        legendgroup=traj_name,
        # legendgrouptitle_text=traj_name,
        line=dict(color=color_sequence[idx]))
    t2 = go.Scatter(x=ts,
                    y=foot_contact_force[:, 1] * 0.8 + 2,
                    mode='lines',
                    name="FL",
                    legendgroup=traj_name,
                    line=dict(color=color_sequence[idx]),
                    showlegend=False)
    t3 = go.Scatter(x=ts,
                    y=foot_contact_force[:, 2] * 0.8 + 1,
                    mode='lines',
                    name="RR",
                    legendgroup=traj_name,
                    line=dict(color=color_sequence[idx]),
                    showlegend=False)
    t4 = go.Scatter(x=ts,
                    y=foot_contact_force[:, 3] * 0.8 + 0,
                    mode='lines',
                    name="RL",
                    legendgroup=traj_name,
                    line=dict(color=color_sequence[idx]),
                    showlegend=False)
    plots.extend([t1, t2, t3, t4])

  foot_contact_fig = go.Figure(data=plots)
  foot_contact_fig.update_layout(
      title="Foot Contact Force (Scaled)",
      xaxis_title="time/s",
      yaxis_title="",
      title_x=0.5,  # This centers the title
      title_y=1.,  # This adjusts the vertical position of the title
      margin=dict(t=20, b=40, l=20, r=20),
      legend=dict(
          x=0,  # This sets the horizontal position of the legend
          y=1,  # This sets the vertical position of the legend
          bgcolor='rgba(255, 255, 255, 0.5)',
          bordercolor='rgba(0, 0, 0, 0.1)',
          borderwidth=1),
      paper_bgcolor='rgba(255, 255, 255, 1.)',
  )
  return [foot_contact_fig]


def create_env_action_div():
  desired_acc_div = html.Div(
      children=[
          html.H3('Environment Action', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="env_action_vx", className="plot"),
                  dcc.Graph(id="env_action_vz", className='plot'),
                  dcc.Graph(id="env_action_angvel_y", className='plot'),
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '33%',
                  # 'overflow': 'hidden'
              }),
          html.Div(
              children=[
                  dcc.Graph(id="env_action_freq", className="plot"),
                  dcc.Graph(id="env_action_front_rx", className='plot'),
                  dcc.Graph(id="env_action_front_rz", className='plot'),
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '33%',
                  # 'overflow': 'hidden'
              }),
          html.Div(
              children=[
                  dcc.Graph(id="env_action_rear_rx", className='plot'),
                  dcc.Graph(id="env_action_rear_rz", className='plot'),
                  dcc.Graph(id="env_action_placeholder", className='plot'),
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '33%',
                  # 'overflow': 'hidden'
              })
      ],
      style={
          'width': '100vw',
          'height': '60vh',
          # 'overflow': 'hidden'
      },
  )
  return desired_acc_div


@app.callback([
    Output('env_action_vx', 'figure'),
    Output('env_action_vz', 'figure'),
    Output('env_action_angvel_y', 'figure'),
    Output('env_action_freq', 'figure'),
    Output('env_action_front_rx', 'figure'),
    Output('env_action_front_rz', 'figure'),
    Output('env_action_rear_rx', 'figure'),
    Output('env_action_rear_rz', 'figure'),
], [Input('traj_name_selector', 'value')])
def update_env_action_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  vx_fig = generate_timeseries_plot(selected_traj_names,
                                    attr_name="env_action",
                                    dim=2,
                                    title="Lin X Vel",
                                    ylabel="m/s")
  vz_fig = generate_timeseries_plot(selected_traj_names,
                                    attr_name="env_action",
                                    dim=4,
                                    title="Lin Z Vel",
                                    ylabel="m/s")
  angvel_y_fig = generate_timeseries_plot(selected_traj_names,
                                          attr_name="env_action",
                                          dim=8,
                                          title="Ang Y Vel",
                                          ylabel="rad/s")
  step_freq_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="env_action",
                                           dim=0,
                                           title="Step Freq",
                                           ylabel="Hz")
  front_rx_fig = generate_timeseries_plot(selected_traj_names,
                                          attr_name="env_action",
                                          dim=-6,
                                          title="Front Residual X",
                                          ylabel="m")
  front_rz_fig = generate_timeseries_plot(selected_traj_names,
                                          attr_name="env_action",
                                          dim=-4,
                                          title="Front Residual Z",
                                          ylabel="m")
  rear_rx_fig = generate_timeseries_plot(selected_traj_names,
                                         attr_name="env_action",
                                         dim=-3,
                                         title="Rear Residual X",
                                         ylabel="m")
  rear_rz_fig = generate_timeseries_plot(selected_traj_names,
                                         attr_name="env_action",
                                         dim=-1,
                                         title="Rear Residual Z",
                                         ylabel="m")
  return (vx_fig, vz_fig, angvel_y_fig, step_freq_fig, front_rx_fig,
          front_rz_fig, rear_rx_fig, rear_rz_fig)


def create_desired_acc_div():
  desired_acc_div = html.Div(
      children=[
          html.H3('Desired Base Acceleration', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="desired_acc_lin_x", className="plot"),
                  dcc.Graph(id="desired_acc_lin_y", className='plot'),
                  dcc.Graph(id="desired_acc_lin_z", className='plot'),
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '50%',
                  # 'overflow': 'hidden'
              }),
          html.Div(
              children=[
                  dcc.Graph(id="desired_acc_ang_x", className="plot"),
                  dcc.Graph(id="desired_acc_ang_y", className='plot'),
                  dcc.Graph(id="desired_acc_ang_z", className='plot'),
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '50%',
                  # 'overflow': 'hidden'
              })
      ],
      style={
          'width': '100vw',
          'height': '40vh',
          # 'overflow': 'hidden'
      },
  )
  return desired_acc_div


@app.callback([
    Output('desired_acc_lin_x', 'figure'),
    Output('desired_acc_lin_y', 'figure'),
    Output('desired_acc_lin_z', 'figure'),
    Output('desired_acc_ang_x', 'figure'),
    Output('desired_acc_ang_y', 'figure'),
    Output('desired_acc_ang_z', 'figure'),
], [Input('traj_name_selector', 'value')])
def update_desired_acc_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  acc_lin_x_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=0,
                                           title="Lin X",
                                           ylabel="m/s^2")
  acc_lin_y_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=1,
                                           title="Lin Y",
                                           ylabel="m/s^2")
  acc_lin_z_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=2,
                                           title="Lin Z",
                                           ylabel="m/s^2")
  acc_ang_x_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=3,
                                           title="Ang X",
                                           ylabel="Rad/s^2")
  acc_ang_y_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=4,
                                           title="Ang Y",
                                           ylabel="Rad/s^2")
  acc_ang_z_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="desired_acc_body_frame",
                                           dim=5,
                                           title="Ang Z",
                                           ylabel="Rad/s^2")
  return (
      acc_lin_x_fig,
      acc_lin_y_fig,
      acc_lin_z_fig,
      acc_ang_x_fig,
      acc_ang_y_fig,
      acc_ang_z_fig,
  )


def create_solved_acc_div():
  desired_acc_div = html.Div(
      children=[
          html.H3('Solved Base Acceleration', style={'height': '4px'}),
          html.Div(
              children=[
                  dcc.Graph(id="solved_acc_lin_x", className="plot"),
                  dcc.Graph(id="solved_acc_lin_y", className='plot'),
                  dcc.Graph(id="solved_acc_lin_z", className='plot')
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '50%',
                  # 'overflow': 'hidden'
              }),
          html.Div(
              children=[
                  dcc.Graph(id="solved_acc_ang_x", className="plot"),
                  dcc.Graph(id="solved_acc_ang_y", className='plot'),
                  dcc.Graph(id="solved_acc_ang_z", className='plot')
              ],
              className='container',
              style={
                  'width': '100vw',
                  'height': '50%',
                  # 'overflow': 'hidden'
              })
      ],
      style={
          'width': '100vw',
          'height': '40vh',
          # 'overflow': 'hidden'
      },
  )
  return desired_acc_div


@app.callback([
    Output('solved_acc_lin_x', 'figure'),
    Output('solved_acc_lin_y', 'figure'),
    Output('solved_acc_lin_z', 'figure'),
    Output('solved_acc_ang_x', 'figure'),
    Output('solved_acc_ang_y', 'figure'),
    Output('solved_acc_ang_z', 'figure')
], [Input('traj_name_selector', 'value')])
def update_solved_acc_figs(selected_traj_names):
  if selected_traj_names is None:
    selected_traj_names = []
  acc_lin_x_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=0,
                                           title="Lin X",
                                           ylabel="m/s^2")
  acc_lin_y_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=1,
                                           title="Lin Y",
                                           ylabel="m/s^2")
  acc_lin_z_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=2,
                                           title="Lin Z",
                                           ylabel="m/s^2")
  acc_ang_x_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=3,
                                           title="Ang X",
                                           ylabel="m/s^2")
  acc_ang_y_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=4,
                                           title="Ang Y",
                                           ylabel="m/s^2")
  acc_ang_z_fig = generate_timeseries_plot(selected_traj_names,
                                           attr_name="solved_acc_body_frame",
                                           dim=5,
                                           title="Ang Z",
                                           ylabel="m/s^2")
  return (
      acc_lin_x_fig,
      acc_lin_y_fig,
      acc_lin_z_fig,
      acc_ang_x_fig,
      acc_ang_y_fig,
      acc_ang_z_fig,
  )


def create_app(app):
  app.layout = html.Div(
      children=[
          html.H1(children='Dog Tracer', style={'textAlign': 'center'}),
          create_upload_div(),
          create_base_position_div(),
          create_base_velocity_div(),
          create_base_orientation_div(),
          create_base_angvel_div(),
          create_foot_contact_div(),
          create_env_action_div(),
          create_desired_acc_div(),
          create_solved_acc_div()
      ],
      style={
          'width': '100vw',
          # 'height': '100vh',
          # 'overflow': 'hidden'
      },
  )
  # app.update_layout()  # This line is required to correctly render the CSS styles
  app.css.append_css({'external_url': 'app.css'})
  return app


if __name__ == '__main__':
  all_data = {}
  depth_images, depth_timestamps = [], []
  create_app(app)
  app.run_server(debug=False)
