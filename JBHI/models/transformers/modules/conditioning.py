import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


#### clustering utils

class TabularKMeans(nn.Module):
    def __init__(self, ncluster=32, nfeat=8, niter=10, return_indices=False, vector_types=None, gamma=None, ckpt_path=None,quantizer_interface=False):
        super().__init__()
        self.ncluster = ncluster
        self.nfeat = nfeat
        self.niter = niter
        self.return_indices = return_indices
        vector_types = vector_types or ['continuous'] * nfeat
        assert len(vector_types) == nfeat, f"Length of vector_types {len(vector_types)} must match nfeat {nfeat}"
        assert all(vt in ['continuous', 'categorical'] for vt in vector_types), "vector_types must be 'continuous' or 'categorical'"
        self.vector_types = vector_types
        self.gamma = gamma  # weight for categorical mismatch part (None => auto)
        self.quantizer_interface = quantizer_interface

        self.register_buffer("C", torch.zeros(ncluster, nfeat))   # centroids (categoricals stored as ints in float tensor)
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
            self.restarted_from_ckpt = True

    # -------- unchanged helpers --------
    def is_initialized(self):
        return self.initialized.item() == 1

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")  # remove weights_only=True; not valid for torch.load
        # allow both raw state_dict and full checkpoints with 'state_dict'
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.load_state_dict(sd, strict=False)
        print(f"Restored TabularKMeans from {path} with init status {self.is_initialized()}")

    # -------- K-Prototypes helpers --------
    def _masks(self, device=None):
        cont_mask = torch.tensor([vt == "continuous" for vt in self.vector_types], dtype=torch.bool, device=device)
        cat_mask  = ~cont_mask
        return cont_mask, cat_mask

    @torch.no_grad()
    def _auto_gamma(self, x, cont_mask, cat_mask):
        """
        Heuristic γ: average per-feature variance on continuous dims (robust-ish).
        If no continuous dims, fall back to 1.0.
        """
        if cont_mask.any():
            xc = x[:, cont_mask].float()
            var = xc.var(dim=0, unbiased=False)
            g = var.mean().clamp(min=1e-8).item()
        else:
            g = 1.0
        return g

    @torch.no_grad()
    def _mixed_distance(self, x, C, cont_mask, cat_mask, gamma):
        """
        x: (N, D) float tensor; categorical columns must be integer-coded stored in x as float/int.
        C: (K, D) centroids (categoricals stored as ints in float)
        returns: (N, K) distances = ||x_cont - C_cont||^2 + γ * Hamming(x_cat, C_cat)
        """
        N, D = x.shape
        K = C.shape[0]
        dist = torch.zeros(N, K, device=x.device, dtype=x.dtype)

        if cont_mask.any():
            x_c = x[:, cont_mask].float()
            C_c = C[:, cont_mask].float()
            # (N,1,Fc) - (1,K,Fc) -> (N,K,Fc) -> sum over last
            d2 = (x_c[:, None, :] - C_c[None, :, :]).pow(2).sum(-1)
            dist += d2

        if cat_mask.any():
            # compare equality on categorical dims; mismatch -> 1.0
            x_cat = x[:, cat_mask].long()
            C_cat = C[:, cat_mask].long()
            # broadcast compare to (N,K,Fm)
            mism = (x_cat[:, None, :] != C_cat[None, :, :]).to(x.dtype)
            hamming = mism.sum(-1)  # (N, K)
            dist += gamma * hamming

        return dist

    @torch.no_grad()
    def _update_centroids(self, x, a, cont_mask, cat_mask):
        """
        Update continuous dims by mean; categorical dims by mode.
        Handles empty clusters by reseeding from random points.
        """
        N, D = x.shape
        K = self.ncluster
        device = x.device
        C_new = torch.empty(K, D, device=device, dtype=x.dtype)

        for k in range(K):
            mask = (a == k)
            if mask.any():
                xs = x[mask]
                # continuous -> mean
                if cont_mask.any():
                    C_new[k, cont_mask] = xs[:, cont_mask].float().mean(0)
                # categorical -> mode (most frequent integer)
                if cat_mask.any():
                    # compute mode per categorical column
                    xcat = xs[:, cat_mask].long()
                    modes = []
                    for j in range(xcat.shape[1]):
                        col = xcat[:, j]
                        # bincount length = 1 + max (safe for 0..M encoding)
                        bc = torch.bincount(col)
                        modes.append(int(torch.argmax(bc)))
                    modes = torch.tensor(modes, device=device, dtype=torch.long)
                    C_new[k, cat_mask] = modes.to(C_new.dtype)
            else:
                # empty cluster -> reseed centroid from a random data point
                ridx = torch.randint(0, N, (1,), device=device)
                C_new[k] = x[ridx]

        return C_new

    # -------- Main: K-Prototypes initializer/trainer --------
    @torch.no_grad()
    def initialize(self, x, niter=None, gamma=None):
        """
        Fit K-Prototypes centroids on mixed-type data.
        Args:
            x: (N, D) tensor; categorical columns must be integer-coded
            niter: number of k-prototypes iterations (defaults to self.niter)
            gamma: weight for categorical mismatch; if None uses _auto_gamma()
        """
        N, D = x.shape
        assert D == self.nfeat, f"Input feature dimension {D} != nfeat {self.nfeat}"
        device = x.device
        x = x.to(device)

        cont_mask, cat_mask = self._masks(device=device)

        # choose gamma
        gamma = float(gamma if gamma is not None else (self.gamma if self.gamma is not None else self._auto_gamma(x, cont_mask, cat_mask)))
        self.gamma = gamma  # remember it

        # init centroids by sampling K points
        C = x[torch.randperm(N, device=device)[:self.ncluster]].clone()

        iters = niter if niter is not None else self.niter
        for _ in range(iters):
            # assign
            dist = self._mixed_distance(x, C, cont_mask, cat_mask, gamma)  # (N,K)
            a = dist.argmin(dim=1)
            # update
            C = self._update_centroids(x, a, cont_mask, cat_mask)

        self.C.copy_(C)
        self.initialized.fill_(1)

    # -------- Forward using mixed distance --------
    def forward(self, x, reverse=False):
        """
        If reverse=False: returns cluster indices (B,)
        If reverse=True: return centroid vectors C[idx] (B, D)
        Uses mixed distance if there are categorical columns; otherwise Euclidean.
        """

        if reverse:
            return self.C[x] # x are indices

        cont_mask, cat_mask = self._masks(device=self.C.device)
        x = x.to(self.C.device, dtype=self.C.dtype)

        # If no categorical dims -> pure k-means
        if not cat_mask.any():
            a = ((x[:, None, :] - self.C[None, :, :])**2).sum(-1).argmin(-1)
            return a

        gamma = float(self.gamma if self.gamma is not None else self._auto_gamma(x, cont_mask, cat_mask))
        dist = self._mixed_distance(x, self.C, cont_mask, cat_mask, gamma)
        a = dist.argmin(dim=1)            
        if self.return_indices:
            if self.quantizer_interface:
                return None, None, a 
            return a               # (B,) # return cluster indices
        else:
            return self.C[a]       # (B, D) # return centroid means

    @staticmethod
    def _encode_categories(values_all: np.ndarray, values_fit: np.ndarray, name: str = "col"):
        """
        Map arbitrary (possibly non-numeric) values to compact integer codes.
        Returns:
            codes_fit: (n_plot,) integer labels for the subsample
            legend_labels: dict {code -> original_value}
        """
        uniques, counts = np.unique(values_all, return_counts=True)
        print(f"[INFO] column '{name}' has {len(uniques)} unique values, e.g. {uniques[:5]} (counts: {counts[:5]})")
        value_to_code = {v: i for i, v in enumerate(uniques)}
        # use vectorized mapping when possible; fallback to list if types are mixed
        try:
            codes_fit = np.array([value_to_code[v] for v in values_fit], dtype=int)
        except Exception:
            codes_fit = np.array(list(map(value_to_code.get, values_fit)), dtype=int)
        legend_labels = {i: v for i, v in enumerate(uniques)}
        return codes_fit, legend_labels

    @staticmethod
    def _fit_reducer(X_fit: np.ndarray, use_umap: bool, random_state: int):
        """
        Fit a 2D reducer (UMAP if available, else PCA) on X_fit and return (reducer, name, Z_fit).
        """
        reducer_name = "UMAP"
        reducer = None
        if use_umap:
            try:
                import umap.umap_ as umap
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=random_state,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric="euclidean",
                    verbose=False,
                )
            except Exception:
                reducer = None

        if reducer is None:
            from sklearn.decomposition import PCA
            reducer_name = "PCA"
            reducer = PCA(n_components=2, random_state=random_state)

        Z_fit = reducer.fit_transform(X_fit)
        return reducer, reducer_name, Z_fit

    @staticmethod
    def _draw_page(
        Z_fit: np.ndarray,
        Z_C: np.ndarray,
        color_labels_fit: np.ndarray,
        *,
        page_title: str,
        reducer_name: str,
        y_km_full: np.ndarray,
        K: int,
        point_size: int,
        point_alpha: float,
        centroid_size: int,
        max_legend_cats: int,
        legend_labels: dict | None,
        show_counts_in_centroids: bool,
    ):
        """
        Draw a single page and return the matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=120)

        uniq = np.unique(color_labels_fit)
        # choose a discrete categorical map (tab10 up to 10 classes, else tab20)
        cmap = cm.get_cmap("tab10" if len(uniq) <= 10 else "tab20")

        # normalize integers to the colormap consistently
        vmin, vmax = int(uniq.min()), int(uniq.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        sc = ax.scatter(
            Z_fit[:, 0], Z_fit[:, 1],
            c=color_labels_fit,
            s=point_size,
            alpha=point_alpha,
            linewidths=0
        )

        # Centroids
        ax.scatter(
            Z_C[:, 0], Z_C[:, 1],
            s=centroid_size,
            marker="X",
            edgecolor="black",
            linewidths=1.0,
            facecolor='none'
        )

        if show_counts_in_centroids:
            counts = np.bincount(y_km_full, minlength=K)
            for k, (zx, zy) in enumerate(Z_C):
                ax.text(zx, zy, f" {k} ({counts[k]})", va="center", ha="left", fontsize=8)

        ax.set_title(f"{page_title} • {reducer_name} projection", fontsize=12)
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.grid(True, alpha=0.15)

        # Legend vs colorbar
        if legend_labels is not None and len(legend_labels) <= max_legend_cats:
            handles = []
            for code, name in legend_labels.items():
                color = sc.cmap(sc.norm(code))   # << exact same color as the scatter
                h = ax.scatter([], [], label=str(name), color=color)
                handles.append(h)
            ax.legend(handles=handles, loc="best", title="category", frameon=True, fontsize=8)
        else:
            cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.01)
            cbar.set_label("label")

        plt.tight_layout()
        return fig

    def save_umap_pages(
        self,
        X: torch.Tensor,                  # (N, B)
        filename: str = "umap_overlays.pdf",
        *,
        col_names: list[str] | None = None,
        use_umap: bool = True,
        random_state: int = 42,
        max_points: int = 50_000,
        point_size: int = 6,
        point_alpha: float = 0.6,
        centroid_size: int = 120,
        max_legend_cats: int = 25,
        show_counts_in_centroids: bool = True,
        auto_bin_continuous: bool = True,
        bin_threshold: int = 50,          # if > this many uniques and float -> bin
        n_bins: int = 10,                 # quantile bins for continuous columns
    ) -> str:
        """
        Make a multi-page PDF:
          - Page 1: colored by k-means cluster id
          - Pages 2..B+1: colored by categories from each input column
        Returns the output filename.
        """
        assert self.is_initialized(), "Model must be initialized/fitted before visualization."

        # Compute cluster labels on device, then bring to CPU
        device = self.C.device
        X = X.to(device, dtype=self.C.dtype)
        with torch.no_grad():
            labels = self(X)  # (N,)

        X_np = X.detach().cpu().float().numpy()        # (N, B)
        y_km = labels.detach().cpu().numpy().astype(int)
        C_np = self.C.detach().cpu().float().numpy()   # (K, B)
        N, B = X_np.shape
        K = self.ncluster

        if col_names is None:
            col_names = [f"col{j}" for j in range(B)]

        # Subsample for speed/clarity (keeps centroids intact)
        rng = np.random.default_rng(random_state)
        if N > max_points:
            idx = rng.choice(N, size=max_points, replace=False)
        else:
            idx = np.arange(N)

        X_fit = X_np[idx]
        y_km_fit = y_km[idx]

        # Fit reducer and transform centroids
        reducer, reducer_name, Z_fit = self._fit_reducer(X_fit, use_umap, random_state)
        Z_C = reducer.transform(C_np)

        # Write PDF
        with PdfPages(filename) as pdf:
            # Page 1: by k-means
            cluster_labels = {i: f"cluster {i}" for i in range(K)}

            fig1 = self._draw_page(
                Z_fit=Z_fit,
                Z_C=Z_C,
                color_labels_fit=y_km_fit,
                page_title="K-Means clusters",
                reducer_name=reducer_name,
                y_km_full=y_km,
                K=K,
                point_size=point_size,
                point_alpha=point_alpha,
                centroid_size=centroid_size,
                max_legend_cats=max_legend_cats,
                legend_labels=cluster_labels,
                show_counts_in_centroids=show_counts_in_centroids,
            )
            pdf.savefig(fig1); plt.close(fig1)

            # Pages per column
            for j in range(B):
                col_all = X_np[:, j]
                col_fit = X_fit[:, j]

                legend_labels = None

                # Optional: auto-bucket continuous columns to avoid thousands of colors
                if auto_bin_continuous and np.issubdtype(col_all.dtype, np.floating) and np.unique(col_all).size > bin_threshold:
                    qs = np.quantile(col_all, np.linspace(0, 1, n_bins + 1))
                    # Use mid-quantile labels 0..n_bins-1
                    col_all_codes = np.digitize(col_all, qs[1:-1], right=True)
                    col_fit_codes = np.digitize(col_fit, qs[1:-1], right=True)
                    # Build legend for bins
                    edges = [f"[{qs[i]:.3g}, {qs[i+1]:.3g})" for i in range(n_bins)]
                    legend_labels = {i: edges[i] for i in range(n_bins)}
                    color_labels_fit = col_fit_codes.astype(int)
                else:
                    # Treat as categorical (works for strings, ints, mixed objects)
                    color_labels_fit, legend_labels = self._encode_categories(col_all, col_fit,name=col_names[j])

                name = col_names[j] if j < len(col_names) else f"col{j}"
                figj = self._draw_page(
                    Z_fit=Z_fit,
                    Z_C=Z_C,
                    color_labels_fit=color_labels_fit,
                    page_title=f"Colored by {name}",
                    reducer_name=reducer_name,
                    y_km_full=y_km,
                    K=K,
                    point_size=point_size,
                    point_alpha=point_alpha,
                    centroid_size=centroid_size,
                    max_legend_cats=max_legend_cats,
                    legend_labels=legend_labels,
                    show_counts_in_centroids=show_counts_in_centroids,
                )
                pdf.savefig(figj); plt.close(figj)

        print(f"[OK] Wrote {B + 1} pages to: {filename}")
        return filename
    
    @torch.no_grad()
    def cluster_stats(self, X: torch.Tensor, as_dataframe: bool = True):
        """
        Compute mean and std per cluster for each feature.
        
        Args:
            self: TabularKMeans model (must be initialized)
            X: torch.Tensor of shape (N, D)
            as_dataframe: if True, return pandas DataFrame; else dict
        
        Returns:
            stats: DataFrame with MultiIndex (cluster, {mean,std}) per feature,
                or dict {cluster_id: {"mean": array, "std": array, "count": int}}
        """
        assert self.is_initialized(), "Model must be initialized/fitted before stats."

        device = self.C.device
        X = X.to(device, dtype=self.C.dtype)
        labels = self(X)

        results = {}
        for k in range(self.ncluster):
            mask = (labels == k)
            if mask.any():
                vals = X[mask]
                mean = vals.mean(0).cpu().numpy()
                std = vals.std(0, unbiased=False).cpu().numpy()
                results[k] = {"mean": mean, "std": std, "count": int(mask.sum().item())}
            else:
                results[k] = {"mean": None, "std": None, "count": 0}

        if as_dataframe:
            # flatten into DataFrame
            rows = []
            for k, d in results.items():
                if d["mean"] is not None:
                    row_mean = {f"f{j}_mean": d["mean"][j] for j in range(len(d["mean"]))}
                    row_std = {f"f{j}_std": d["std"][j] for j in range(len(d["std"]))}
                    row = {"cluster": k, "count": d["count"], **row_mean, **row_std}
                else:
                    row = {"cluster": k, "count": 0}
                rows.append(row)
            df = pd.DataFrame(rows).set_index("cluster")
            return df
        else:
            return results