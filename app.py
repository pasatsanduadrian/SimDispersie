import os
from dotenv import load_dotenv
from flask import Flask, request, render_template, send_file
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import osmnx as ox
import contextily as ctx
from tqdm import tqdm
import base64
from datetime import datetime
from pyngrok import ngrok
import threading

# CONSTANTĂ GLOBALĂ pentru dimensiune imagine
FIGSIZE = (9, 9)  # Modifică aici dacă vrei să schimbi global dimensiunea

load_dotenv()
app = Flask(__name__)

# -------------- UTILITARE MODEL DISPERSIE -----------------
def calc_sigmas(CATEGORY, x1):
    """Calculate sigma_y and sigma_z for Pasquill stability CATEGORY."""
    x = np.abs(x1)
    a = np.zeros(np.shape(x))
    b = np.zeros(np.shape(x))
    c = np.zeros(np.shape(x))
    d = np.zeros(np.shape(x))
    if CATEGORY == 1:
        ind = np.where((x < 100.) & (x > 0.)); a[ind] = 122.800; b[ind] = 0.94470
        ind = np.where((x >= 100.) & (x < 150.)); a[ind] = 158.080; b[ind] = 1.05420
        ind = np.where((x >= 150.) & (x < 200.)); a[ind] = 170.220; b[ind] = 1.09320
        ind = np.where((x >= 200.) & (x < 250.)); a[ind] = 179.520; b[ind] = 1.12620
        ind = np.where((x >= 250.) & (x < 300.)); a[ind] = 217.410; b[ind] = 1.26440
        ind = np.where((x >= 300.) & (x < 400.)); a[ind] = 258.89; b[ind] = 1.40940
        ind = np.where((x >= 400.) & (x < 500.)); a[ind] = 346.75; b[ind] = 1.7283
        ind = np.where((x >= 500.) & (x < 3110.)); a[ind] = 453.85; b[ind] = 2.1166
        ind = np.where(x >= 3110.); a[ind] = 453.85; b[ind] = 2.1166
        c[:] = 24.1670; d[:] = 2.5334
    elif CATEGORY == 2:
        ind = np.where((x < 200.) & (x > 0.)); a[ind] = 90.673; b[ind] = 0.93198
        ind = np.where((x >= 200.) & (x < 400.)); a[ind] = 98.483; b[ind] = 0.98332
        ind = np.where(x >= 400.); a[ind] = 109.3; b[ind] = 1.09710
        c[:] = 18.3330; d[:] = 1.8096
    elif CATEGORY == 3:
        a[:] = 61.141; b[:] = 0.91465; c[:] = 12.5; d[:] = 1.0857
    elif CATEGORY == 4:
        ind = np.where((x < 300.) & (x > 0.)); a[ind] = 34.459; b[ind] = 0.86974
        ind = np.where((x >= 300.) & (x < 1000.)); a[ind] = 32.093; b[ind] = 0.81066
        ind = np.where((x >= 1000.) & (x < 3000.)); a[ind] = 32.093; b[ind] = 0.64403
        ind = np.where((x >= 3000.) & (x < 10000.)); a[ind] = 33.504; b[ind] = 0.60486
        ind = np.where((x >= 10000.) & (x < 30000.)); a[ind] = 36.650; b[ind] = 0.56589
        ind = np.where(x >= 30000.); a[ind] = 44.053; b[ind] = 0.51179
        c[:] = 8.3330; d[:] = 0.72382
    elif CATEGORY == 5:
        ind = np.where((x < 100.) & (x > 0.)); a[ind] = 24.26; b[ind] = 0.83660
        ind = np.where((x >= 100.) & (x < 300.)); a[ind] = 23.331; b[ind] = 0.81956
        ind = np.where((x >= 300.) & (x < 1000.)); a[ind] = 21.628; b[ind] = 0.75660
        ind = np.where((x >= 1000.) & (x < 2000.)); a[ind] = 21.628; b[ind] = 0.63077
        ind = np.where((x >= 2000.) & (x < 4000.)); a[ind] = 22.534; b[ind] = 0.57154
        ind = np.where((x >= 4000.) & (x < 10000.)); a[ind] = 24.703; b[ind] = 0.50527
        ind = np.where((x >= 10000.) & (x < 20000.)); a[ind] = 26.970; b[ind] = 0.46713
        ind = np.where((x >= 20000.) & (x < 40000.)); a[ind] = 35.420; b[ind] = 0.37615
        ind = np.where(x >= 40000.); a[ind] = 47.618; b[ind] = 0.29592
        c[:] = 6.25; d[:] = 0.54287
    elif CATEGORY == 6:
        ind = np.where((x < 200.) & (x > 0.)); a[ind] = 15.209; b[ind] = 0.81558
        ind = np.where((x >= 200.) & (x < 700.)); a[ind] = 14.457; b[ind] = 0.78407
        ind = np.where((x >= 700.) & (x < 1000.)); a[ind] = 13.953; b[ind] = 0.68465
        ind = np.where((x >= 1000.) & (x < 2000.)); a[ind] = 13.953; b[ind] = 0.63227
        ind = np.where((x >= 2000.) & (x < 3000.)); a[ind] = 14.823; b[ind] = 0.54503
        ind = np.where((x >= 3000.) & (x < 7000.)); a[ind] = 16.187; b[ind] = 0.46490
        ind = np.where((x >= 7000.) & (x < 15000.)); a[ind] = 17.836; b[ind] = 0.41507
        ind = np.where((x >= 15000.) & (x < 30000.)); a[ind] = 22.651; b[ind] = 0.32681
        ind = np.where((x >= 30000.) & (x < 60000.)); a[ind] = 27.074; b[ind] = 0.27436
        ind = np.where(x >= 60000.); a[ind] = 34.219; b[ind] = 0.21716
        c[:] = 4.1667; d[:] = 0.36191
    else:
        raise ValueError("Unknown stability category.")
    sig_z = a * (x / 1000.)**b
    sig_z[x > 0][sig_z[x > 0] > 5000.] = 5000.
    theta = 0.017453293 * (c - d * np.log(np.abs(x + 1e-15) / 1000.))
    sig_y = 465.11628 * x / 1000. * np.tan(theta)
    return sig_y, sig_z

def gauss_func(Q, u, dir1, x, y, z, xs, ys, H_eff, Dy, Dz, STABILITY):
    """Gaussian plume concentration at receptors."""
    u1 = u
    x1 = x - xs
    y1 = y - ys
    wx = u1 * np.sin((dir1 - 180.) * np.pi / 180.)
    wy = u1 * np.cos((dir1 - 180.) * np.pi / 180.)
    dot_product = wx * x1 + wy * y1
    magnitudes = u1 * np.sqrt(x1**2 + y1**2)
    subtended = np.arccos(dot_product / (magnitudes + 1e-15))
    downwind = np.cos(subtended) * magnitudes
    crosswind = np.sin(subtended) * magnitudes
    sig_y, sig_z = calc_sigmas(STABILITY, downwind)
    C = np.zeros(x.shape)
    ind = downwind > 0
    exp_part = np.exp(-crosswind[ind]**2 / (2 * sig_y[ind]**2 + 1e-15))
    exp_z = (np.exp(- (z[ind] - H_eff)**2 / (2 * sig_z[ind]**2 + 1e-15)) +
             np.exp(- (z[ind] + H_eff)**2 / (2 * sig_z[ind]**2 + 1e-15)))
    C[ind] = Q / (2 * np.pi * u1 * sig_y[ind] * sig_z[ind] + 1e-15) * exp_part * exp_z
    return C

def calculate_plume_rise(T_stack, T_ambient, wind_speed, flow_rate, stack_diameter=0.5):
    """Estimate plume rise (m) based on stack and ambient conditions."""
    T_stack_K = T_stack + 273.15
    T_ambient_K = T_ambient + 273.15
    delta_T = T_stack_K - T_ambient_K
    delta_H = stack_diameter * (flow_rate / (wind_speed + 1e-15))**0.25 * (1 + (delta_T / (T_stack_K + 1e-15)))
    return delta_H

def estimate_stability(row):
    """Infer Pasquill stability class from weather row."""
    if row['WS'] > 4.5: return 6
    elif row['WS'] > 3.5: return 5
    elif row['RAD'] > 200 and row['TC'] > 20: return 4
    elif row['RAD'] > 120 and row['TC'] > 15: return 3
    elif row['RAD'] > 50: return 2
    elif row['RH'] > 75: return 1
    else: return 1

# "YlGnBu", "BuGn", "viridis"
def overlay_on_map(local_x, local_y, C1, image_data, extent_range, figsize=FIGSIZE, plume_cmap='BuGn'):
    """Plot plume concentrations on top of network image."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    local_extent = [-extent_range, extent_range, -extent_range, extent_range]
    ax.imshow(image_data, extent=local_extent, origin='lower', alpha=1.0)
    C1_max = np.max(C1, axis=2) * 1e6
    C1_max = np.nan_to_num(C1_max)
    max_val = np.max(C1_max)
    min_val = np.min(C1_max[C1_max > 0]) if np.any(C1_max > 0) else 0
    if max_val == min_val:
        thresholds = np.linspace(min_val, max_val + 1e-9, 7)
    else:
        thresholds = np.linspace(min_val, max_val, 7)
    thresholds = np.unique(thresholds)
    if len(thresholds) < 2:
        thresholds = np.array([min_val, max_val + 1e-9])
    cmap = plt.get_cmap(plume_cmap)
    cs = ax.contourf(local_x, local_y, C1_max, levels=thresholds, cmap=cmap, alpha=0.65, extend='neither')
    cbar = fig.colorbar(cs, ticks=thresholds, fraction=0.025, pad=0.02)
    cbar.set_label(r'$\mu g/m^3$')
    ax.scatter(0, 0, color='red', s=80, edgecolor='k', zorder=5, label='Emission Source')
    ax.set_xlim(local_extent[0], local_extent[1])
    ax.set_ylim(local_extent[2], local_extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m, local)')
    ax.set_ylabel('y (m, local)')
    ax.set_title('Dispersion Plume Overlaid on Local Street Map')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig, C1_max, min_val, max_val

def run_dispersion_model_local(center_lon, center_lat, street_network_image,
                              emission_source_x, emission_source_y, flow_rate,
                              H, T_stack, crs_proj, extent_range, size_in_pixels):
    """Simulate hourly dispersion using a sample CSV of meteorological data."""
    x_range = np.linspace(emission_source_x - extent_range, emission_source_x + extent_range, size_in_pixels)
    y_range = np.linspace(emission_source_y - extent_range, emission_source_y + extent_range, size_in_pixels)
    x, y = np.meshgrid(x_range, y_range)
    local_x = x - emission_source_x
    local_y = y - emission_source_y
    df = pd.read_csv("data/sample_weather.csv", parse_dates=["Time"])
    df['Stability'] = df.apply(estimate_stability, axis=1)
    df['Q'] = df['S1'] / 1e6
    y_dim, x_dim = x.shape
    C1 = np.zeros((y_dim, x_dim, 24))
    for i in tqdm(range(24), desc="Simulating dispersion"):
        wind_dir = df.loc[i, 'WD']
        wind_speed = df.loc[i, 'WS']
        T_ambient = df.loc[i, 'TC']
        stability = df.loc[i, 'Stability']
        H_eff = H + calculate_plume_rise(T_stack, T_ambient, wind_speed, flow_rate)
        C = gauss_func(df.loc[i, 'Q'], wind_speed, wind_dir, x, y, np.zeros((y_dim, x_dim)),
                       emission_source_x, emission_source_y, H_eff, 10, 10, stability)
        mask = (
            (x < emission_source_x - extent_range) | (x > emission_source_x + extent_range) |
            (y < emission_source_y - extent_range) | (y > emission_source_y + extent_range)
        )
        C[mask] = np.nan
        C1[:, :, i] = C
    fig_overlay, C1_max, min_conc, max_conc = overlay_on_map(
        local_x, local_y, C1, street_network_image, extent_range, figsize=FIGSIZE
    )
    return fig_overlay, min_conc, max_conc

def generate_network_basemap(center_lat, center_lon, dist=2500, dxy=10, dpi=100, figsize=FIGSIZE):
    """Return basemap image centered at given location."""
    import pyproj
    crs_3857 = "EPSG:3857"
    G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='drive')
    G_proj = ox.project_graph(G, to_crs=crs_3857)
    crs_proj = G_proj.graph['crs']
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_3857, always_xy=True)
    emission_source_x, emission_source_y = transformer.transform(center_lon, center_lat)
    extent_range = dist
    size_in_pixels = int((2 * extent_range) / dxy)
    fig, ax = ox.plot_graph(
        G_proj, bgcolor='none', edge_color='#003366', node_size=0,
        edge_linewidth=0.5, show=False, close=False, figsize=figsize
    )
    ax.set_xlim([emission_source_x - extent_range, emission_source_x + extent_range])
    ax.set_ylim([emission_source_y - extent_range, emission_source_y + extent_range])

    try:
        # Basemap color (OpenStreetMap Mapnik are toate detaliile color, nu doar străzi)
        ctx.add_basemap(
            ax, 
            crs=crs_proj, 
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom=15, 
            reset_extent=False
        )
    except Exception as e:
        print(f"Eroare la contextily: {e}")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer._renderer)[..., :3]
    path = "/tmp/network_basemap.png"
    fig.savefig(path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return image_array, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels, path


def generate_network_only(center_lat, center_lon, dist=2500, dxy=10, dpi=100, figsize=FIGSIZE):
    """Return network image without basemap."""
    G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='drive')
    node_id = ox.nearest_nodes(G, center_lon, center_lat)
    G_proj = ox.project_graph(G, to_crs="EPSG:3857")
    crs_proj = G_proj.graph['crs']
    emission_source_x = G_proj.nodes[node_id]['x']
    emission_source_y = G_proj.nodes[node_id]['y']
    extent_range = dist
    size_in_pixels = int((2 * extent_range) / dxy)
    fig, ax = ox.plot_graph(
        G_proj, bgcolor='white', edge_color='#003366', node_size=0,
        edge_linewidth=0.5, show=False, close=False, figsize=figsize
    )
    ax.set_xlim([emission_source_x - extent_range, emission_source_x + extent_range])
    ax.set_ylim([emission_source_y - extent_range, emission_source_y + extent_range])
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer._renderer)[..., :3]
    plt.close(fig)
    return image_array, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels


@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        latitude = float(request.form.get("latitude", 44.4399))
        longitude = float(request.form.get("longitude", 26.0550))
        radius = float(request.form.get("radius", 2500))
        flow_rate = float(request.form.get("flow_rate", 5))
        H = float(request.form.get("H", 25))
        T_stack = float(request.form.get("T_stack", 150))
        dxy = 10
        dpi = 100

        # Harta cu străzi + basemap color
        network_img, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels, network_path = generate_network_basemap(
            latitude, longitude, dist=radius, dxy=dxy, dpi=dpi, figsize=FIGSIZE
        )
        # Doar străzi pentru pluma
        street_network_image, _, _, _, _, _ = generate_network_only(
            latitude, longitude, dist=radius, dxy=dxy, dpi=dpi, figsize=FIGSIZE
        )

        fig_overlay, min_conc, max_conc = run_dispersion_model_local(
            longitude, latitude, street_network_image,
            emission_source_x, emission_source_y, flow_rate, H, T_stack, crs_proj, extent_range, size_in_pixels
        )
        overlay_path = "/tmp/plume.png"
        fig_overlay.savefig(overlay_path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_overlay)

        with open(network_path, "rb") as imgfile:
            network_bytes = imgfile.read()
        with open(overlay_path, "rb") as imgfile:
            overlay_bytes = imgfile.read()

        network_b64 = base64.b64encode(network_bytes).decode()
        overlay_b64 = base64.b64encode(overlay_bytes).decode()

        metadate = [
            ("Data generării", datetime.now().strftime("%d/%m/%Y %H:%M")),
            ("Coordonate centru", f"{latitude}, {longitude}"),
            ("Rază analiză", f"{radius} m"),
            ("Flow rate", f"{flow_rate} m³/s"),
            ("Înălțime sursă", f"{H} m"),
            ("Temperatură gaze", f"{T_stack} °C"),
            ("Grid step", f"{dxy} m"),
            ("Emission source (x, y)", f"{emission_source_x:.2f}, {emission_source_y:.2f}"),
            ("Max conc", f"{max_conc:.2f} μg/m³"),
            ("Min conc (>0)", f"{min_conc:.2f} μg/m³")
        ]
        with open("/tmp/last_network.png", "wb") as f:
            f.write(network_bytes)
        with open("/tmp/last_overlay.png", "wb") as f:
            f.write(overlay_bytes)
        import pyproj, folium
        trans = pyproj.Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
        lon_min, lat_min = trans.transform(emission_source_x - extent_range, emission_source_y - extent_range)
        lon_max, lat_max = trans.transform(emission_source_x + extent_range, emission_source_y + extent_range)
        m = folium.Map(location=[latitude, longitude], zoom_start=13)
        folium.raster_layers.ImageOverlay(
            overlay_path,
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            opacity=0.6
        ).add_to(m)
        m.save("/tmp/interactive_map.html")
        return render_template(
            "result.html",
            network_b64=network_b64,
            overlay_b64=overlay_b64,
            metadata=metadate
        )
    except Exception as e:
        return render_template(
            "result.html", network_b64="", overlay_b64="", metadata=[("Eroare", str(e))]
        )

@app.route("/image/network")
def img_network():
    return send_file("/tmp/last_network.png", mimetype="image/png")

@app.route("/image/overlay")
def img_overlay():
    return send_file("/tmp/last_overlay.png", mimetype="image/png")

@app.route("/map")
def map_view():
    return send_file("/tmp/interactive_map.html", mimetype="text/html")

def run_with_ngrok(app, port=5015):
    """Run Flask app exposing it via ngrok if token is set."""
    port = int(os.getenv("PORT", port))
    ngrok_token = os.getenv("NGROK_TOKEN")
    ngrok_hostname = os.getenv("NGROK_HOSTNAME")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        try:
            if ngrok_hostname:
                public_url = ngrok.connect(port, hostname=ngrok_hostname)
            else:
                public_url = ngrok.connect(port)
        except Exception as e:
            print("Eroare la conectarea cu hostname static:", e)
            public_url = ngrok.connect(port)
        print(f"\n*** LINK STATIC NGROK: {public_url.public_url}\n")
    else:
        print("Warning: nu ai setat token ngrok. Aplicația va fi disponibilă doar local.")

    def run_flask():
        app.run(host='0.0.0.0', port=port, debug=False)

    thread = threading.Thread(target=run_flask)
    thread.daemon = True
    thread.start()
    thread.join()

if __name__ == "__main__":
    run_with_ngrok(app, port=int(os.getenv("PORT", 5015)))
