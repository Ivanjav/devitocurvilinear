import numpy as np

def stretch_eta_sinh(n_eta, H, beta=0.0):
    s = np.linspace(0.0, 1.0, n_eta)
    if beta == 0.0:
        return H * s
    return H * np.sinh(beta*s) / np.sinh(beta)

def meshgrid_from_topo(x_top, y_top, n_xi=512, n_eta=256, H=4.0,
                       max_iter=15, tol=1e-10, damping=0.5, mfit=1024, beta=0.0,
                       verbose=False):
    # ----- 0) Periodic, uniformly sampled topography over one period -----
    x_top0 = np.asarray(x_top, float)
    y_top0 = np.asarray(y_top, float)
    dx = x_top0[1]-x_top0[0]
    nextent = 0.1*x_top0.size
    n_xi0 = n_xi
    n_xi = int(n_xi0*1.1)
    # n_eta = int(n_eta*1.1)
    x_top = np.hstack([x_top0,x_top0[-1]+nextent*dx])  # Extend to avoid issues at the boundary
    y_top = np.hstack([y_top0, y_top0[0]])
    order = np.argsort(x_top)
    x_top = x_top[order]; y_top = y_top[order]
    L = x_top[-1] - x_top[0]
    if L <= 0:
        raise ValueError("x_top must span a positive length.")
    x0 = x_top[0]
    x_top = x_top - x0

    Mfit = int(max(mfit, 4 * n_xi))
    x_fit = np.linspace(0.0, L, Mfit, endpoint=False)

    # map to [0,L) and collapse duplicates for monotone interp
    x_mod = np.mod(x_top, L)
    idx = np.argsort(x_mod)
    x_mod = x_mod[idx]; y_sorted = y_top[idx]
    dupe = np.diff(x_mod, prepend=x_mod[0]-1.0) == 0
    x1 = x_mod[~dupe]; y1 = y_sorted[~dupe]

    y_fit = np.interp(x_fit, x1, y1, period=L)
    y_mean = y_fit.mean()
    y0 = y_fit - y_mean
    Y = np.fft.rfft(y0) / Mfit               # k=0..Mfit//2
    N = (Mfit // 2) - 1                      # exclude Nyquist
    A_full =  2 * Y[1:N+1].real
    B_full = -2 * Y[1:N+1].imag
    a0 = y_mean

    # ----- 1) w-plane grid and harmonic cap -----
    S = L / (2.0 * np.pi)
    xi = np.linspace(0, 2*np.pi, n_xi, endpoint=False)
    eta = stretch_eta_sinh(n_eta, 2*np.pi*H/L, beta=beta)  # NOTE: scaled H in w-plane

    Ncap = min(N, (n_xi // 2) - 1) if n_xi >= 4 else 1
    A = A_full[:Ncap].copy()
    B = B_full[:Ncap].copy()

    # ----- 2) Precompute ξ and η factors once (key speedup) -----
    # For n=1..Ncap: exp(i n w) = e^{-n η} (cos nξ + i sin nξ)
    n = np.arange(1, Ncap+1, dtype=float)        # (N,)
    cos_nxi = np.cos(np.outer(n, xi))            # (N, n_xi)
    sin_nxi = np.sin(np.outer(n, xi))            # (N, n_xi)
    decay = np.exp(-np.outer(n, eta))            # (N, n_eta)
    decay_T = decay.T                            # (n_eta, N)

    # Helpers to evaluate z on entire grid using only GEMMs
    # Real/imag contributions:
    # Re add:  Σ e^{-nη} [ B_n cos(nξ) - A_n sin(nξ) ]
    # Im add:  Σ e^{-nη} [ A_n cos(nξ) + B_n sin(nξ) ]
    def eval_full_grid(A, B, a0):
        # (n_eta, n_xi)
        term_Bcos = decay_T @ (B[:, None] * cos_nxi)
        term_Asin = decay_T @ (A[:, None] * sin_nxi)
        term_Acos = decay_T @ (A[:, None] * cos_nxi)
        term_Bsin = decay_T @ (B[:, None] * sin_nxi)
        X = (S * xi)[None, :] + (term_Bcos - term_Asin)
        Y = (a0 + S * eta)[:, None] + (term_Acos + term_Bsin)
        return X, Y

    # Fast 1D evaluation at top (η=0 ⇒ decay=1), avoids 2D work inside iterations
    ones = np.ones_like(n)
    def eval_top(A, B, a0):
        # y_top_est(ξ) = a0 + Σ [ A_n cos(nξ) + B_n sin(nξ) ]
        return a0 + (A @ cos_nxi) + (B @ sin_nxi)

    # periodic y(x) sampler
    def y_periodic(xquery):
        return np.interp(np.mod(xquery, L), x_fit, y_fit, period=L)

    # ----- 3) Boundary-consistency refinement -----
    # Start with current coeffs
    for it in range(max_iter):
        # Top boundary mapping with current coefficients:
        # z_top(ξ) = b + S ξ + Σ c_n e^{i n ξ};  Im(b)=a0
        y_top_est = eval_top(A, B, a0)

        # Map ξ-column abscissas onto physical x (top row, η=0):
        # Real part at top: x_top(ξ) = S ξ + Σ [ B_n cos(nξ) - A_n sin(nξ) ]
        x_top_est = (S * xi) + (B @ cos_nxi) - (A @ sin_nxi)

        # Target topography samples at mapped abscissas
        y_tgt = y_periodic(np.mod(x_top_est, L))

        # New Fourier coefficients from the samples on the ξ grid
        a0_new = y_tgt.mean()
        Ynew = np.fft.rfft(y_tgt - a0_new) / n_xi
        A_new =  2 * Ynew[1:Ncap+1].real
        B_new = -2 * Ynew[1:Ncap+1].imag

        # Damped update
        A = (1.0 - damping) * A + damping * A_new
        B = (1.0 - damping) * B + damping * B_new
        a0 = (1.0 - damping) * a0 + damping * a0_new

        # Residual after update (cheap check on top)
        y_top_new = eval_top(A, B, a0)
        err = np.max(np.abs(y_top_new - y_tgt))
        if verbose:
            print(f"iter {it}: top max err = {err:.3e}")
        if err < tol:
            break

    # ----- 4) Final mapping on full grid (η×ξ) -----
    X, Y = eval_full_grid(A, B, a0)

    #Wrap columns to [0, L) using the top row
    x_top_raw = X[0, :]
    cols_right = np.where(x_top_raw > L)[0]
    if cols_right.size:
        X[:, cols_right] -= L
    x_top_raw = X[0, :]
    cols_left = np.where(x_top_raw < 0)[0]
    if cols_left.size:
        X[:, cols_left] += L

    x_top_wrapped = np.mod(X[0, :], L)
    j0 = int(np.argmin(x_top_wrapped))
    X = np.roll(X, -j0, axis=1)
    Y = np.roll(Y, -j0, axis=1)

    mask = (X[0, :] > x_top0[0]) & (X[0, :] < x_top0[-1])
    ind = np.flatnonzero(mask)   # equivalente a np.where(mask)[0]

    X = X[:,ind]
    Y = Y[:,ind]    
    # Interppolate X and Y to original n_xi0 size
    from scipy.interpolate import interp1d
    xi_new = np.linspace(0, 2*np.pi, n_xi0, endpoint=False)
    X_interp = np.zeros((n_eta, n_xi0))
    Y_interp = np.zeros((n_eta, n_xi0))
    xi_old = np.linspace(0, 2*np.pi, X.shape[1], endpoint=False)
    for i in range(n_eta):
        fX = interp1d(xi_old, X[i,:], kind='cubic', fill_value="extrapolate")
        fY = interp1d(xi_old, Y[i,:], kind='cubic', fill_value="extrapolate")
        X_interp[i,:] = fX(xi_new)
        Y_interp[i,:] = fY(xi_new)
    return X_interp, Y_interp


from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

def mapping_velocity(vp, dx, dz, X, Z):
    """
    Map the velocity model 'vp' defined on a regular (x,z) grid
    onto the curvilinear grid given by coordinate fields X, Z.

    Faster alternative to griddata: uses RegularGridInterpolator
    and a single nearest pass for out-of-bounds fill.
    """
    # 1) Replace zeros (invalids) on the regular grid by nearest valid value (fast)
    vp = np.asarray(vp)
    mask_invalid = vp <= 0
    if np.any(mask_invalid):
        # distance_transform_edt gives indices to nearest False (valid) pixel
        # return_indices=True returns a tuple of index arrays you can use to pick values
        nearest_idx = distance_transform_edt(mask_invalid, return_distances=False, return_indices=True)
        vp_filled = vp.copy()
        vp_filled[mask_invalid] = vp[tuple(nearest_idx)][mask_invalid]
    else:
        vp_filled = vp

    # 2) Build fast interpolators on the regular grid
    # Axes in physical coordinates
    x_axis = np.arange(vp_filled.shape[0]) * dx
    z_axis = np.arange(vp_filled.shape[1]) * dz

    # Linear interpolation first
    rgi_linear = RegularGridInterpolator(
        (x_axis, z_axis), vp_filled.astype(np.float32, copy=False),
        method='linear', bounds_error=False, fill_value=np.nan
    )

    # Evaluate on curvilinear coordinates
    pts = np.column_stack((X.ravel(), Z.ravel()))
    out = rgi_linear(pts).reshape(X.shape)

    # 3) Fill any NaNs (outside convex hull) with a single nearest-neighbor pass
    if np.isnan(out).any():
        rgi_nearest = RegularGridInterpolator(
            (x_axis, z_axis), vp_filled.astype(np.float32, copy=False),
            method='nearest', bounds_error=False, fill_value=None
        )
        nan_mask = np.isnan(out)
        if nan_mask.any():
            out[nan_mask] = rgi_nearest(pts[nan_mask.ravel()])

    return out

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

class CurviMap:
    """
    Fast mapping between computational (Xi,Eta) and physical (X,Z) coordinates
    for general curvilinear grids (no tensor-product assumption).
    
    Once initialized, c2p() and p2c() are very fast since the Delaunay
    triangulations are built only once.
    """
    def __init__(self, Xi, Eta, X, Z, dtype=np.float32):
        # store grids
        self.Xi  = np.asarray(Xi, dtype)
        self.Eta = np.asarray(Eta, dtype)
        self.X   = np.asarray(X, dtype)
        self.Z   = np.asarray(Z, dtype)

        if not (self.Xi.shape == self.Eta.shape == self.X.shape == self.Z.shape):
            raise ValueError("Xi, Eta, X, Z must all have the same 2D shape.")

        # --- computational -> physical (c2p) ---
        pts_c = np.column_stack((self.Xi.ravel(), self.Eta.ravel()))
        self._X_lin = LinearNDInterpolator(pts_c, self.X.ravel(), fill_value=np.nan)
        self._Z_lin = LinearNDInterpolator(pts_c, self.Z.ravel(), fill_value=np.nan)
        self._X_nn  = NearestNDInterpolator(pts_c, self.X.ravel())
        self._Z_nn  = NearestNDInterpolator(pts_c, self.Z.ravel())

        # --- physical -> computational (p2c) ---
        pts_p = np.column_stack((self.X.ravel(), self.Z.ravel()))
        self._Xi_lin  = LinearNDInterpolator(pts_p, self.Xi.ravel(), fill_value=np.nan)
        self._Eta_lin = LinearNDInterpolator(pts_p, self.Eta.ravel(), fill_value=np.nan)
        self._Xi_nn   = NearestNDInterpolator(pts_p, self.Xi.ravel())
        self._Eta_nn  = NearestNDInterpolator(pts_p, self.Eta.ravel())

    # ---------- forward mapping ----------
    def c2p(self, xi, eta):
        """Map (xi, eta) -> (x, z)"""
        xi = np.asarray(xi, self.X.dtype)
        eta = np.asarray(eta, self.Z.dtype)
        pts = np.column_stack((xi.ravel(), eta.ravel()))

        x = self._X_lin(pts).reshape(xi.shape)
        z = self._Z_lin(pts).reshape(eta.shape)

        # fill NaNs with nearest neighbor
        mask = np.isnan(x)
        if mask.any():
            x[mask] = self._X_nn(pts[mask.ravel()]).reshape(x.shape)[mask]
        mask = np.isnan(z)
        if mask.any():
            z[mask] = self._Z_nn(pts[mask.ravel()]).reshape(z.shape)[mask]
        return x, z

    # ---------- inverse mapping ----------
    def p2c(self, x, z):
        """Map (x, z) -> (xi, eta)"""
        x = np.asarray(x, self.X.dtype)
        z = np.asarray(z, self.Z.dtype)
        pts = np.column_stack((x.ravel(), z.ravel()))

        xi = self._Xi_lin(pts).reshape(x.shape)
        eta = self._Eta_lin(pts).reshape(z.shape)

        # fill NaNs with nearest neighbor
        mask = np.isnan(xi)
        if mask.any():
            xi[mask] = self._Xi_nn(pts[mask.ravel()])
        mask = np.isnan(eta)
        if mask.any():
            eta[mask] = self._Eta_nn(pts[mask.ravel()])
        return xi, eta
    
    def vel_c2p(self, v_xi, v_eta, x, z, eps=1e-12):
        """
        Simple 2D transform of (v_xi, v_eta) -> (v_x, v_z) at physical points (x,z).
        Assumes Xi,Eta are tensor-product grids (regular in each param).
        """
        import numpy as np
        from scipy.interpolate import RegularGridInterpolator

        # 1) param axes (eta: rows, xi: cols)
        eta = self.Eta[:, 0]
        xi  = self.Xi[0, :]
        deta = eta[1] - eta[0]
        dxi  = xi[1]  - xi[0]

        # 2) metric terms on grid
        x_eta, x_xi = np.gradient(self.X, deta, dxi, edge_order=2)
        z_eta, z_xi = np.gradient(self.Z, deta, dxi, edge_order=2)
        h_xi  = np.sqrt(x_xi**2 + z_xi**2)
        h_eta = np.sqrt(x_eta**2 + z_eta**2)

        # 3) interpolators on (eta, xi)
        def RGI(A):
            return RegularGridInterpolator((eta, xi), A, bounds_error=False, fill_value=None)

        I_x_xi  = RGI(x_xi);  I_x_eta = RGI(x_eta)
        I_z_xi  = RGI(z_xi);  I_z_eta = RGI(z_eta)
        I_h_xi  = RGI(h_xi);  I_h_eta = RGI(h_eta)

        # 4) map (x,z) -> (xi,eta) with the class’ inverse
        xi_r, eta_r = self.p2c(x, z)
        pts = np.column_stack((eta_r, xi_r))

        x_xi_r  = I_x_xi(pts);  x_eta_r = I_x_eta(pts)
        z_xi_r  = I_z_xi(pts);  z_eta_r = I_z_eta(pts)
        h_xi_r  = I_h_xi(pts);  h_eta_r = I_h_eta(pts)

        # 5) components
        v_x = v_xi * (x_xi_r / h_xi_r) + v_eta * (x_eta_r / h_eta_r)
        v_z = v_xi * (z_xi_r / h_xi_r) + v_eta * (z_eta_r / h_eta_r)
        return v_x, v_z



# def mapping_c2p(xi, eta, Xi, Eta, X, Z):
#     """Mapping from computational to physical domain"""
#     from scipy.interpolate import griddata
#     x = griddata(points=(Xi.ravel(), Eta.ravel()), values=X.ravel(), xi=(xi, eta), method='linear')
#     z = griddata(points=(Xi.ravel(), Eta.ravel()), values=Z.ravel(), xi=(xi, eta), method='linear')
#     return x, z

# def mapping_p2c(x,z,X,Z,Xi,Eta):
#     """Mapping from physical to computational domain"""
#     from scipy.interpolate import griddata
#     xi = griddata(points=(X.ravel(), Z.ravel()), values=Xi.ravel(), xi=(x, z), method='linear')
#     eta = griddata(points=(X.ravel(), Z.ravel()), values=Eta.ravel(), xi=(x, z), method='linear')
#     return xi, eta
# import numpy as np

# def stretch_eta_sinh(n_eta, H, beta=0.0):
#     s = np.linspace(0.0, 1.0, n_eta)
#     if beta == 0:
#         return H*s
#     return H * np.sinh(beta*s) / np.sinh(beta)


# def meshgrid_from_topo(x_top, y_top, n_xi=512, n_eta=256, H=4.0,
#                                   max_iter=15, tol=1e-10, damping=0.5, mfit=1024, beta =0.0):
#     """
#     Build a 2D conformal curvilinear grid (X,Z) for a periodic topography y(x).

#     Parameters
#     ----------
#     x_top, y_top : 1D arrays
#         Horizontal positions and topography elevations (same length). Assumed periodic over L.
#     n_xi, n_eta : int
#         Number of points along (xi, eta). xi runs along the surface, eta is vertical depth.
#     H : float
#         Depth (in the same units as y_top) of the computational domain (eta in [0, H]).
#     max_iter : int
#         Max iterations for boundary-consistency refinement.
#     tol : float
#         Convergence tolerance on the top boundary mismatch.
#     damping : float in (0,1]
#         Damping for coefficient updates (0.5 is robust).
#     mfit : int
#         FFT size for fitting Fourier coefficients of the topography (use power of two).

#     Returns
#     -------
#     X, Z : 2D arrays (n_eta, n_xi)
#         Physical coordinates of the conformal grid.
#     """

#     # ----- 0) Prepare periodic, uniformly sampled topography over one period -----
#     x_top = np.asarray(x_top, dtype=float)
#     y_top = np.asarray(y_top, dtype=float)
#     # sort by x and shift to [0, L)
#     order = np.argsort(x_top)
#     x_top = x_top[order]
#     y_top = y_top[order]
#     L = x_top[-1] - x_top[0]
#     if L <= 0:
#         raise ValueError("x_top must span a positive length.")

#     # shift origin to 0
#     x0 = x_top[0]
#     x_top = x_top - x0
#     # enforce periodic endpoints by wrapping interpolation domain
#     Mfit = int(max(mfit, 4 * n_xi))  # enough spectral resolution
#     x_fit = np.linspace(0.0, L, Mfit, endpoint=False)

#     # periodic interpolation of y onto uniform x_fit
#     # map x_top to [0,L), build a periodic extension (two periods) and interpolate
#     x_mod = (x_top % L)
#     idx = np.argsort(x_mod)
#     x_mod = x_mod[idx]
#     y_sorted = y_top[idx]
#     # build one-period monotone samples for interp
#     # collapse duplicates if any
#     dupe = np.diff(x_mod, prepend=x_mod[0]-1.0) == 0
#     x1 = x_mod[~dupe]
#     y1 = y_sorted[~dupe]
#     # interpolate on [0,L)
#     y_fit = np.interp(x_fit, x1, y1, period=L)

#     # remove mean, get Fourier coefficients (real series)
#     y_mean = y_fit.mean()
#     y0 = y_fit - y_mean
#     Y = np.fft.rfft(y0) / Mfit                     # k = 0..Mfit/2
#     N = (Mfit // 2) - 1                            # exclude Nyquist
#     # real Fourier series: y(x) = a0 + sum A_n cos(2π n x/L) + B_n sin(...)
#     A =  2 * Y[1:N+1].real
#     B = -2 * Y[1:N+1].imag
#     a0 = y_mean

#     # ----- 1) Set up w-plane grid (xi, eta) and initial analytic-series coeffs -----
#     S = L / (2.0 * np.pi)  # scale
#     xi  = np.linspace(0, 2*np.pi, n_xi, endpoint=False)
#     #eta = np.linspace(0, H,        n_eta)
#     eta = stretch_eta_sinh(n_eta, 2*np.pi*H/L, beta=beta)
#     XI, ETA = np.meshgrid(xi, eta, indexing='xy')
#     w = XI + 1j * ETA

#     # cap harmonics to xi Nyquist
#     Ncap = min(N, (n_xi // 2) - 1) if n_xi >= 4 else 1
#     A = A[:Ncap]
#     B = B[:Ncap]
#     c = B + 1j * A           # complex coefficients for exp(i n w)
#     b = 1j * a0              # imaginary constant produces vertical shift

#     # helper: periodic interpolation of y_fit at arbitrary x in R
#     def y_periodic(xquery):
#         xq = np.mod(xquery, L)
#         return np.interp(xq, x_fit, y_fit, period=L)

#     # ----- 2) Boundary-consistency refinement: enforce y_top(ξ) = y(x_top(ξ)) -----
#     for it in range(max_iter):
#         # map with current coefficients
#         z = b + S * w
#         if Ncap > 0:
#             n = np.arange(1, Ncap + 1)[:, None, None]  # (Ncap,1,1)
#             z = z + np.sum(c[:, None, None] * np.exp(1j * n * w), axis=0)

#         xg = np.real(z)
#         yg = np.imag(z)

#         # top row samples
#         x_top_map = np.mod(xg[0, :], L)
#         y_top_map = yg[0, :]

#         # target y at mapped abscissas
#         y_tgt = y_periodic(x_top_map)

#         # fit new Fourier coeffs directly on xi grid points (x_top_map corresponds to xi samples)
#         a0_new = y_tgt.mean()
#         Ynew = np.fft.rfft(y_tgt - a0_new) / n_xi
#         A_new =  2 * Ynew[1:Ncap+1].real
#         B_new = -2 * Ynew[1:Ncap+1].imag
#         c_new = B_new + 1j * A_new
#         b_new = 1j * a0_new

#         # error and update (damped for robustness)
#         err = np.max(np.abs(y_top_map - y_tgt))

#         c = (1.0 - damping) * c + damping * c_new
#         b = (1.0 - damping) * b + damping * b_new

#         # quick convergence check with updated top only
#         z_top = b + S * xi
#         if Ncap > 0:
#             n1 = np.arange(1, Ncap + 1)[:, None]   # (Ncap,1)
#             z_top = z_top + np.sum(c[:, None] * np.exp(1j * n1 * xi), axis=0)
#         y_top_est = np.imag(z_top)
#         err_new = np.max(np.abs(y_top_est - y_tgt))
#         print(f"iter {it}: top max err = {err_new:.3e}")
#         if err_new < tol:
#             break

#     # ----- 3) Final mapping on full grid -----
#     z = b + S * w
#     if Ncap > 0:
#         n = np.arange(1, Ncap + 1)[:, None, None]
#         z = z + np.sum(c[:, None, None] * np.exp(1j * n * w), axis=0)

#     xg = np.real(z)
#     yg = np.imag(z)

#     # --- 1) Wrap columns whose top xg is outside [0, L) ---
#     x_top_raw = xg[0, :]

#     # wrap right overshoot: x > L  → subtract L
#     cols_right = np.where(x_top_raw > L)[0]
#     if cols_right.size:
#         xg[:, cols_right] -= L

#     # wrap left overshoot: x < 0   → add L
#     x_top_raw = xg[0, :]  # refresh
#     cols_left = np.where(x_top_raw < 0)[0]
#     if cols_left.size:
#         xg[:, cols_left] += L


#     x_top_wrapped = np.mod(xg[0, :], L)
#     j0 = int(np.argmin(x_top_wrapped))      # column to become first
#     xg = np.roll(xg, -j0, axis=1)
#     yg = np.roll(yg, -j0, axis=1)


#     return xg, yg
