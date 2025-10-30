import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV ===
try:
    df = pd.read_csv('live_ra_above.csv')
except FileNotFoundError:
    print("Error: 'live_session_down.csv' not found in the current directory.")
    exit()

# === Check expected columns ===
expected_cols = ['Timestamp', 'Acceleration Magnitude', 'Gyro Magnitude', 'Roll', 'Pitch']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print(f"Error: Missing columns in CSV: {missing}")
    exit()

# === Prepare data ===
df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp']).sort_values(by='Timestamp')

plot_items = [
    ('Acceleration Magnitude', 'Acceleration Magnitude vs Time'),
    ('Gyro Magnitude', 'Gyro Magnitude vs Time'),
    ('Roll', 'Roll vs Time'),
    ('Pitch', 'Pitch vs Time'),
]

fig, axes = plt.subplots(len(plot_items), 1, figsize=(10, 10), sharex=True)

# === Plot all ===
for ax, (col, title) in zip(axes, plot_items):
    ax.plot(df['Timestamp'], df[col], linewidth=1)
    ax.set_ylabel(col)
    ax.set_title(title)
    ax.grid(True)

axes[-1].set_xlabel('Timestamp')
plt.tight_layout()

# === Enable scroll zoom + drag pan for all subplots ===
pressed = False
x0 = None
xlim0 = None

# Store initial full range for % zoom calculation
x_full_min = df['Timestamp'].min()
x_full_max = df['Timestamp'].max()
x_full_range = x_full_max - x_full_min

# Add zoom text overlay (top-right)
zoom_text = fig.text(
    0.95, 0.97, "Zoom: 100%", ha="right", va="top", fontsize=10, color="blue", alpha=0.8
)

def update_zoom_label(ax):
    """Update the zoom percentage label."""
    x_min, x_max = ax.get_xlim()
    current_range = x_max - x_min
    zoom_percent = (x_full_range / current_range) * 100
    zoom_text.set_text(f"Zoom: {zoom_percent:.1f}%")

def on_scroll(event):
    """Zoom all plots together with scroll wheel."""
    scale_factor = 1.2
    for ax in axes:
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        xdata = event.xdata
        if xdata is None:
            continue
        if event.button == 'up':  # zoom in
            new_range = x_range / scale_factor
        elif event.button == 'down':  # zoom out
            new_range = x_range * scale_factor
        else:
            continue
        left = xdata - (xdata - x_min) * (new_range / x_range)
        right = left + new_range
        ax.set_xlim(left, right)
    update_zoom_label(axes[0])
    fig.canvas.draw_idle()

def on_press(event):
    """Start dragging (panning)."""
    global pressed, x0, xlim0
    if event.button == 1:  # left mouse
        pressed = True
        x0 = event.xdata
        xlim0 = [ax.get_xlim() for ax in axes]

def on_release(event):
    """Stop dragging."""
    global pressed
    pressed = False

def on_motion(event):
    """Handle drag motion."""
    global pressed, x0, xlim0
    if not pressed or event.xdata is None or x0 is None:
        return
    dx = event.xdata - x0
    for i, ax in enumerate(axes):
        x_min, x_max = xlim0[i]
        ax.set_xlim(x_min - dx, x_max - dx)
    update_zoom_label(axes[0])
    fig.canvas.draw_idle()

# === Connect events ===
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

print("✅ Scroll mouse to zoom, click & drag to pan (all subplots move together).")
print("🔍 Zoom percentage shown in top-right corner of window.")

plt.show()
