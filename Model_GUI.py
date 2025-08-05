import tkinter as tk
from tkinter import filedialog, messagebox
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -- Model loading & inference functions --

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, activation="relu", dropout=0.0, normalization="none", weight_norm=False, output_activation="none"):
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "prelu": nn.PReLU(),
            "tanh": nn.Tanh(),
        }
        act = acts.get(activation, nn.ReLU())
        layers = []
        prev = in_dim
        for h in hidden:
            lin = nn.Linear(prev, h)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            layers.append(lin)
            if normalization == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif normalization == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        if output_activation == "relu":
            self.out_act = nn.ReLU()
        elif output_activation == "softplus":
            self.out_act = nn.Softplus()
        else:
            self.out_act = nn.Identity()
    def forward(self, x):
        return self.out_act(self.net(x))

def load_model_and_scalers(model_path, scalers_path, cfg):
    # Load model architecture
    model = MLP(
        in_dim=len(cfg["feature_names"]),
        out_dim=len(cfg["target_names"]),
        hidden=cfg["hidden_sizes"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        normalization=cfg["normalization"],
        weight_norm=cfg["weight_norm"],
        output_activation=cfg["output_activation"]
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    # Load scalers
    with open(scalers_path, "r") as f:
        scalers = json.load(f)
    return model, scalers

def predict_sweep(model, scalers, cfg, features, frame_range):
    # features: dict of fixed features
    # frame_range: (start, end)
    n_frames = frame_range[1] - frame_range[0] + 1
    X = np.zeros((n_frames, len(cfg["feature_names"])), dtype=np.float32)
    for i, frame in enumerate(range(frame_range[0], frame_range[1] + 1)):
        X[i, :] = [
            features["chamber_pressure"],
            features["injection_pressure"],
            features["injection_duration"],
            features["control_backpressure"]
        ]
    # Standardize
    if scalers["x"] is not None:
        mu = np.array(scalers["x"]["mu"])
        sigma = np.array(scalers["x"]["sigma"])
        X = (X - mu) / sigma
    # Predict
    with torch.no_grad():
        preds = model(torch.from_numpy(X.astype(np.float32))).numpy()
    # Inverse target scale
    if scalers["y"] is not None:
        mu = np.array(scalers["y"]["mu"])
        sigma = np.array(scalers["y"]["sigma"])
        preds = preds * sigma + mu
    # preds: shape (n_frames, 3)
    return preds

# -- Main GUI --

class InferenceApp:
    def __init__(self, root):
        self.root = root
        root.title("Spray MLP Inference")
        
        # Configuration (must match training)
        self.cfg = {
            "feature_names": ["chamber_pressure", "injection_pressure", "injection_duration", "control_backpressure"],
            "target_names": ["cone_angle", "penetration_index", "area"],
            "hidden_sizes": [32, 32, 16],
            "activation": "tahn",
            "dropout": 0.5,
            "normalization": "layer",
            "weight_norm": False,
            "output_activation": "none"
        }
        
        # Paths
        tk.Label(root, text="Model (.pt):").grid(row=0, column=0, sticky="e")
        self.model_entry = tk.Entry(root, width=40)
        self.model_entry.grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_model).grid(row=0, column=2)
        
        tk.Label(root, text="Scalers (.json):").grid(row=1, column=0, sticky="e")
        self.scalers_entry = tk.Entry(root, width=40)
        self.scalers_entry.grid(row=1, column=1)
        tk.Button(root, text="Browse", command=self.browse_scalers).grid(row=1, column=2)
        
        # Feature inputs
        self.entries = {}
        params = ["chamber_pressure", "injection_pressure", "injection_duration", "control_backpressure"]
        for i, p in enumerate(params):
            tk.Label(root, text=p + ":").grid(row=2+i, column=0, sticky="e")
            e = tk.Entry(root)
            e.grid(row=2+i, column=1)
            self.entries[p] = e
        
        # Frame range
        tk.Label(root, text="Start Frame:").grid(row=6, column=0, sticky="e")
        self.start_entry = tk.Entry(root)
        self.start_entry.grid(row=6, column=1, sticky="w")
        tk.Label(root, text="End Frame:").grid(row=7, column=0, sticky="e")
        self.end_entry = tk.Entry(root)
        self.end_entry.grid(row=7, column=1, sticky="w")
        
        tk.Button(root, text="Add Run", command=self.add_run).grid(row=8, column=1)
        
        # Matplotlib figures for the three outputs
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(6, 8))
        self.ax1.set_title("Cone Angle vs Frame")
        self.ax1.set_xlabel("Frame")
        self.ax1.set_ylabel("Cone Angle")
        self.ax2.set_title("Penetration vs Frame")
        self.ax2.set_xlabel("Frame")
        self.ax2.set_ylabel("Penetration")
        self.ax3.set_title("Area vs Frame")
        self.ax3.set_xlabel("Frame")
        self.ax3.set_ylabel("Area")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=9)
        
        self.runs = []
    
    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt")])
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)
    
    def browse_scalers(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self.scalers_entry.delete(0, tk.END)
            self.scalers_entry.insert(0, path)
    
    def add_run(self):
        try:
            model_path = self.model_entry.get()
            scalers_path = self.scalers_entry.get()
            if not model_path or not scalers_path:
                raise ValueError("Model and scalers paths must be specified.")
            # Load model once
            if not hasattr(self, 'model'):
                self.model, self.scalers = load_model_and_scalers(model_path, scalers_path, self.cfg)
            
            features = {p: float(self.entries[p].get()) for p in self.entries}
            start = int(self.start_entry.get())
            end = int(self.end_entry.get())
            if end < start:
                raise ValueError("End frame must be >= start frame")
            
            preds = predict_sweep(self.model, self.scalers, self.cfg, features, (start, end))
            frames = np.arange(start, end+1)
            
            # Plot new lines
            self.ax1.plot(frames, preds[:,0], label=f"run{len(self.runs)+1}")
            self.ax2.plot(frames, preds[:,1], label=f"run{len(self.runs)+1}")
            self.ax3.plot(frames, preds[:,2], label=f"run{len(self.runs)+1}")
            
            for ax in (self.ax1, self.ax2, self.ax3):
                ax.legend()
            
            self.canvas.draw()
            self.runs.append((features, (start, end)))
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()

