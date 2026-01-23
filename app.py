import dash
from dash import html, dcc, Input, Output, State, callback_context
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import butter, sosfiltfilt, filtfilt, savgol_filter
import numpy as np
from src.process_data.parser import Reader
import sys
import numpy as np
from datetime import datetime
import re
import os, base64
sys.path.append("/Users/chinmay/test_code")

UPLOAD_DIR = "data/uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = dash.Dash(__name__)
server = app.server

# ---------- Helpers ----------

def natural_key(name):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', name)]

def stft(signal, fs):
    signal -= np.mean(signal)
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals) / N
    freqs = np.fft.rfftfreq(N, d=1/fs)
    pos = freqs >= 0
    return freqs, fft_mag, pos

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = fs / 2
    sos = butter(order, [lowcut/nyq, highcut/nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)

def heartSignal(signal, fs, lowcut, highcut):
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return bandpass_filter(signal, fs, lowcut, highcut)

def respSignal(signal, fs, lowcut, highcut):
    sos = butter(4, [lowcut/(fs/2), highcut/(fs/2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)

def ypad2(a, b, pad=0.15):
    m = max(np.max(np.abs(a)), np.max(np.abs(b)))
    return [-m*(1+pad), m*(1+pad)]

def downsample(x, fs, new_fs):
    factor = fs // new_fs

    cutoff = 0.45 * new_fs  # safe margin
    b, a = butter(6, cutoff / (fs/2))
    x_filt = filtfilt(b, a, x)

    return x_filt[::factor]

# ---------- Layout ----------

app.layout = html.Div(id="page-wrapper", className="center-layout", children=[

    html.Div(className="controls", children=[
        html.Div(className="toolbar", children=[
            html.Button("◀ Previous", id="prev-btn"),
            dcc.Upload(id="upload-files", children=html.Button("Upload"), multiple=True),
            dcc.Dropdown(id="file-selector", placeholder="Select file"),
            html.Button("Generate", id="generate-btn"),

            html.Div(className="input-strip", children=[
                html.Label("HR"),
                dcc.Input(id="hr-start", type="number", value=5),
                html.Span("–"),
                dcc.Input(id="hr-end", type="number", value=20),
                html.Label("RR"),
                dcc.Input(id="rr-start", type="number", value=0.1),
                html.Span("–"),
                dcc.Input(id="rr-end", type="number", value=0.7),
            ]),

            html.Div(id="current-file-label", className="file-label"),
            html.Div(id="header-time-label", className="header-time"),

            html.Div(className="spacer"),
            html.Button("Next ▶", id="next-btn")
        ])
    ]),

    dcc.Graph(id="mega-plot", style={"flex": "1", "display": "none", "height": "100%"})
])

# ---------- Upload ----------

@app.callback(
    Output("file-selector", "options"),
    Input("upload-files", "contents"),
    State("upload-files", "filename"),
    prevent_initial_call=True
)
def save_files(contents, names):
    if not contents:
        return []

    if isinstance(names, str):
        contents, names = [contents], [names]

    # ---- SORT HERE ----
    pairs = sorted(
        zip(names, contents),
        key=lambda x: natural_key(x[0])
    )
    names, contents = zip(*pairs)
    # -------------------

    saved = []
    for c, n in zip(contents, names):
        data = base64.b64decode(c.split(",")[1])
        path = os.path.join(UPLOAD_DIR, n)
        with open(path, "wb") as f:
            f.write(data)
        saved.append(path)

    return [{"label": os.path.basename(p), "value": p} for p in saved]


# ---------- File Navigation ----------

@app.callback(
    Output("file-selector", "value"),
    Input("next-btn", "n_clicks"),
    Input("prev-btn", "n_clicks"),
    State("file-selector", "options"),
    State("file-selector", "value"),
    prevent_initial_call=True
)
def navigate_files(next_clicks, prev_clicks, options, current):

    ctx = callback_context
    if not ctx.triggered or not options:
        return current

    values = [o["value"] for o in options]
    if current not in values:
        return values[0]

    idx = values.index(current)
    button = ctx.triggered[0]["prop_id"].split(".")[0]

    if button == "next-btn":
        return values[(idx + 1) % len(values)]
    elif button == "prev-btn":
        return values[(idx - 1) % len(values)]

    return current

# ---------- Controls ----------

@app.callback(
    Output("page-wrapper", "className"),
    Output("mega-plot", "style"),
    Input("generate-btn", "n_clicks"),
    prevent_initial_call=True
)
def move_controls(_):
    return "top-layout", {"flex": "1", "display": "block", "height": "100%"}


# ---------- Current File Label ----------

@app.callback(
    Output("current-file-label", "children"),
    Input("file-selector", "value")
)
def show_current_filename(path):
    if not path:
        return ""
    return f"File: {os.path.basename(path)}"


# ---------- Plot ----------

@app.callback(
    Output("mega-plot", "figure"),
    Output("header-time-label", "children"),
    Input("generate-btn", "n_clicks"),
    Input("file-selector", "value"),
    State("hr-start", "value"),
    State("hr-end", "value"),
    State("rr-start", "value"),
    State("rr-end", "value"),
    prevent_initial_call=True
)
def update_plots(_, filepath, hr_s, hr_e, rr_s, rr_e):

    # ch1, ch2, _, _ = Reader(filepath).read_channels()
    ch1, ch2, gain_list, last_epoch, header_meta_data, fsr_values, fsr_time, calculate_packet_loss, total_frames = Reader(filepath).read_channels()
    fs = round(((calculate_packet_loss + total_frames) * 100) / ((last_epoch[-1] - last_epoch[0]) / 1000))
    ch1 = ch2
    ch1 = downsample(np.array(ch1), fs, 250)
    ch2 = downsample(np.array(ch2), fs, 250)
    # ch1 = ch1 - savgol_filter(np.array(ch1), 5000, 2)
    # ch2 = ch2 - savgol_filter(np.array(ch2), 5000, 2)

    header_time = int(header_meta_data["Timestamp"])
    header_time_text = (
        "Start Time: "
        + datetime.fromtimestamp(header_time).strftime("%Y-%m-%d %H:%M:%S")
    )
    fs = 250
    t = np.arange(len(ch1)) / fs

    h1, h2 = heartSignal(ch1, fs, hr_s, hr_e), heartSignal(ch2, fs, hr_s, hr_e)
    r1, r2 = respSignal(ch1, fs, rr_s, rr_e), respSignal(ch2, fs, rr_s, rr_e)
    f1, m1, p1 = stft(ch1, fs)
    f2, m2, p2 = stft(ch2, fs)

    fig = make_subplots(
        rows=4,
        cols=2,
        vertical_spacing=0.05,
        horizontal_spacing=0.03,
        subplot_titles=[
            "Channel-1", "Channel-2"
        ]
    )

    fig.update_annotations(font_size=15)

    # ----- Row 1 : Raw BCG -----
    fig.add_trace(go.Scatter(x=t, y=ch1, line=dict(color="#0041c3"), opacity=0.8), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=ch2, line=dict(color="#0041c3"), opacity=0.8), 1, 2)

    # ----- Row 2 : Heart band -----
    fig.add_trace(go.Scatter(x=t, y=h1, line=dict(color="#ED2939"), opacity=1), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=h2, line=dict(color="#ED2939"), opacity=1), 2, 2)

    # ----- Row 3 : Resp band -----
    fig.add_trace(go.Scatter(x=t, y=r1, line=dict(color="#ff7f0e")), 3, 1)
    fig.add_trace(go.Scatter(x=t, y=r2, line=dict(color="#ff7f0e")), 3, 2)

    # ----- Row 4 : FFT -----
    fig.add_trace(go.Scatter(x=f1[(p1 >= 0) & (p1 <= 20)], y=m1[(p1 >= 0) & (p1 <= 20)], line=dict(color="#8c564b")), 4, 1)
    fig.add_trace(go.Scatter(x=f2[(p2 >= 0) & (p2 <= 20)], y=m2[(p2 >= 0) & (p2 <= 20)], line=dict(color="#8c564b")), 4, 2)
    # fig.add_trace(go.Scatter(x=f1[p1], y=m1[p1]), 4, 1)
    # fig.add_trace(go.Scatter(x=f2[p2], y=m2[p2]), 4, 2)
    # print(f1[(p1 >= 0) & (p1 <= 20)])
    fig.layout.xaxis2.matches = 'x'
    fig.layout.xaxis3.matches = 'x'
    fig.layout.xaxis5.matches = 'x'
    fig.layout.xaxis4.matches = 'x2'
    fig.layout.xaxis6.matches = 'x2'

    # ----- Axis titles -----

    fig.update_xaxes(title_text="Time (s)", title_standoff=5, row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", title_standoff=5, row=3, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", title_standoff=5, row=4, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", title_standoff=5, row=4, col=2)

    fig.update_xaxes(range=[0,20], fixedrange=False, row=4, col=1)
    fig.update_xaxes(range=[0,20], fixedrange=False, row=4, col=2)

    yl = [ypad2(ch1,ch2), ypad2(h1,h2), ypad2(r1,r2),
          [0, max(np.max(m1[p1]), np.max(m2[p2]))*1.15]]
    title_y = ["Raw BCG", "Heart", "Resp", "FFT Magnitude"]
    for i in range(4):
        fig.update_yaxes(title_text=title_y[i], range=yl[i], fixedrange=False, row=i+1, col=1)
        fig.update_yaxes(range=yl[i], fixedrange=False, row=i+1, col=2)

    fig.update_layout(
        autosize=True,
        plot_bgcolor="#EAEAF2",
        paper_bgcolor="#F5F5F7",
        margin=dict(l=20, r=30, t=20, b=30),
        showlegend=False
    )

    return fig, header_time_text

if __name__ == "__main__":
    app.run(debug=True)
