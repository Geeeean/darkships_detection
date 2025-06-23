import numpy as np
from pyproj import Proj
import scipy


from hydrophone import Hydrophone


class CoordinateHandler:
    def __init__(self, hydrophones: list[Hydrophone], ref_lat=None, ref_lon=None):
        """Initialize coordinate handler with reference point"""
        self.hydrophones = hydrophones
        if ref_lat is None:
            ref_lat = hydrophones[0].coord.latitude

        if ref_lon is None:
            ref_lon = hydrophones[0].coord.longitude

        self._init_utm_projection(ref_lat, ref_lon)

    def _init_utm_projection(self, ref_lat, ref_lon):
        """Initialize UTM projection based on reference location"""
        # Determine UTM zone based on the reference point
        utm_zone = int((ref_lon + 180) / 6) + 1
        south = ref_lat < 0

        # Create projection string for UTM
        proj_str = f"+proj=utm +zone={utm_zone} {'+south' if south else ''} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

        # Create transformer
        self.utm_proj = Proj(proj_str)

    def geo_to_utm(self, lats, lons):
        """
        Convert geographic coordinates (lat, lon) to UTM coordinates (easting, northing)
        Works with individual coordinates or arrays
        Returns UTM coordinates in meters
        """
        if isinstance(lats, (list, np.ndarray)) and isinstance(
            lons, (list, np.ndarray)
        ):
            return self.utm_proj(lons, lats)
        else:
            return self.utm_proj(lons, lats)

    def utm_to_geo(self, eastings, northings):
        """
        Convert UTM coordinates back to geographic coordinates
        Works with individual coordinates or arrays
        """
        return self.utm_proj(eastings, northings, inverse=True)

    def get_utm_coords(self):
        # Get UTM coordinates of hydrophones
        lats = [h.coord.latitude for h in self.hydrophones]
        lons = [h.coord.longitude for h in self.hydrophones]
        eastings, northings = self.geo_to_utm(lats, lons)
        s = np.array(
            list(zip(eastings, northings))
        )  # s represents hydrophone positions

        return s


class WCLLocalizer:
    def __init__(self, hydrophones: list[Hydrophone]):
        self.hydrophones = hydrophones

    def weighted_centroid_localize(self) -> tuple[float, float]:
        deltas = [h.compute_pressure_delta() for h in self.hydrophones]
        delta_tot = sum(deltas)

        if delta_tot == 0:
            return (0.0, 0.0)

        weighted_lat = sum(
            h.coord.latitude * d for h, d in zip(self.hydrophones, deltas)
        )
        weighted_lon = sum(
            h.coord.longitude * d for h, d in zip(self.hydrophones, deltas)
        )

        return (weighted_lat / delta_tot, weighted_lon / delta_tot)


class TDOALocalizer(CoordinateHandler):
    def __init__(self, hydrophones: list[Hydrophone], sound_speed=1500):
        super().__init__(hydrophones=hydrophones)
        self.v = sound_speed

    def tdoa_localize(self, v_water=None):
        """
        Implement TDOA localization using the algorithm from the paper [1].
        Returns the estimated position in geographic coordinates (lat, lon).
        """
        # Use class-level sound speed if none provided
        if v_water is None:
            v_water = self.v

        s = self.get_utm_coords()

        # First hydrophone (reference)
        s1 = s[0]
        x1, y1 = s1

        # Second hydrophone
        if len(s) < 2:
            raise ValueError("Need at least two hydrophones for TDOA")

        s2 = s[1]
        x2, y2 = s2

        # Calculate time differences (TDOA)
        # Reference time is from first hydrophone
        t1 = self.hydrophones[0].observed_pressure[-1]["toa"]

        # Pre-calculate tau_2 (time difference between second and first hydrophone)
        tau_2 = self.hydrophones[1].observed_pressure[-1]["toa"] - t1

        A = []
        D = []

        # Process hydrophones from index 2 onwards
        for i in range(2, len(self.hydrophones)):
            xi, yi = s[i]

            # Calculate time difference between current hydrophone and reference
            tau_i = self.hydrophones[i].observed_pressure[-1]["toa"] - t1

            # Guard against division by zero
            if abs(tau_i) < 1e-10 or abs(tau_2) < 1e-10:
                continue

            # Computing A_i, B_i, D_i as per equations (15)-(17)
            Ai = (1 / (v_water * tau_i)) * (-2 * x1 + 2 * xi) - (
                1 / (v_water * tau_2)
            ) * (-2 * x1 + 2 * x2)

            Bi = (1 / (v_water * tau_i)) * (-2 * y1 + 2 * yi) - (
                1 / (v_water * tau_2)
            ) * (-2 * y1 + 2 * y2)

            Di = (
                v_water * tau_i
                - v_water * tau_2
                + (1 / (v_water * tau_i)) * (x1**2 + y1**2 - xi**2 - yi**2)
                - (1 / (v_water * tau_2)) * (x1**2 + y1**2 - x2**2 - y2**2)
            )

            A.append([Ai, Bi])
            D.append(-Di)

        # Check if we have enough measurements to compute position
        if len(A) == 0:
            raise ValueError("Not enough valid measurements to compute position")

        # Convert to numpy arrays
        A = np.array(A)
        D = np.array(D)

        # Moore-Penrose pseudoinverse to solve A @ [x, y] = D as in equation (18)
        position_utm = np.linalg.pinv(A) @ D

        # Convert UTM coordinates back to geographic coordinates
        lon, lat = self.utm_to_geo(position_utm[0], position_utm[1])

        return (lat, lon)


class TMMLocalizer(CoordinateHandler):
    def __init__(self, hydrophones: list[Hydrophone], sound_speed=1500):
        super().__init__(hydrophones=hydrophones)
        self.v = sound_speed

    def _compute_distances(self):
        # Get TOA values and calculate distance differences
        toa_values = [h.observed_pressure[-1]["toa"] for h in self.hydrophones]
        ref_toa = toa_values[0]
        tdoas = [toa - ref_toa for toa in toa_values]
        distances = [tdoa * self.v for tdoa in tdoas]  # Convert TDOA to distances

        return distances

    def _find_initial_point(self):
        """
        Algorithm 2 from the paper: Find an optimal initial point for the MM algorithm
        Reference: equations (28)-(29)
        """

        s = self.get_utm_coords()
        distances = self._compute_distances()

        # Step 1: Find index k for which f(s_k) is minimum
        # Equation (8) from paper: f(x) = sum_i (||x - s_i|| - d_i)^2
        min_f_val = float("inf")
        min_idx = 0

        for k in range(len(s)):
            f_val = 0
            for i in range(len(s)):
                # d_i is distances[i], ||s_k - s_i|| is Euclidean distance
                dist = np.linalg.norm(s[k] - s[i])
                f_val += (dist - distances[i]) ** 2

            if f_val < min_f_val:
                min_f_val = f_val
                min_idx = k

        # Step 2: Calculate gradient g_k(s_k)
        # Equation (29) from paper
        g_k = np.zeros(2)
        for i in range(len(s)):
            if i != min_idx:
                s_k_minus_s_i = s[min_idx] - s[i]
                dist = np.linalg.norm(s_k_minus_s_i)

                if dist > 1e-10:  # Avoid division by zero
                    g_k += 2 * (dist - distances[i]) * s_k_minus_s_i / dist

        # Step 3: Determine direction vector v_0
        # Use equation in Algorithm 2 from paper
        if np.linalg.norm(g_k) > 1e-10:
            v_0 = -g_k / np.linalg.norm(g_k)
        else:
            # If gradient is zero or very small, use arbitrary unit vector
            v_0 = np.array([1.0, 0.0])

        # Step 4: Search for step size t with binary search
        t = 1.0  # Initial value
        x_0 = s[min_idx] + t * v_0

        # Function to evaluate f(x) at position x
        def evaluate_f(x):
            f_val = 0
            for i in range(len(s)):
                dist = np.linalg.norm(x - s[i])
                f_val += (dist - distances[i]) ** 2
            return f_val

        # f(s_k) at starting point
        f_s_k = evaluate_f(s[min_idx])

        # Binary search to find optimal t
        max_iterations = 20
        for iter in range(max_iterations):
            x_0 = s[min_idx] + t * v_0
            f_x0 = evaluate_f(x_0)

            # Check condition in Algorithm 2: f(s_k + t*v_0) < f(s_k)
            if f_x0 < f_s_k:
                break

            # Reduce t and try again
            t /= 2.0

            # If t becomes too small, use s_k as initial point
            if t < 1e-8:
                x_0 = s[min_idx]
                break

        # If we reach maximum iterations without finding a valid point
        if iter == max_iterations - 1:
            x_0 = s[min_idx]

        return x_0

    def tmm_localize(self, max_iterations=100, tolerance=1e-8):
        # Step 1: Initial x est
        x_init = self._find_initial_point()
        # print(x_init)

        return self.tmm_iteration(x_init, max_iterations, tolerance)

    def tmm_localize_sr_ls_init(self, max_iterations=100, tolerance=1e-8):
        est = Core.sr_ls_localization(self.hydrophones)
        utm_est = self.geo_to_utm(est[0], est[1])
        x_init = [utm_est[0], utm_est[1]]

        return self.tmm_iteration(x_init, max_iterations, tolerance)

    def tmm_iteration(self, x_init, max_iterations, tolerance):
        """
        Algorithm 3 from the paper: T-MM underwater acoustic-based location algorithm.
        Reference: equations (28)-(29)
        """

        s = self.get_utm_coords()
        tdoa_diffs = self._compute_distances()  # Δd_i = v_water * τ_i (d_i - d_1)

        d1_init = np.linalg.norm(x_init - s[0])
        di_fixed = [d1_init + tdoa_diffs[i] for i in range(len(s))]

        def compute_objective(x):
            return sum(
                (np.linalg.norm(x - s[i]) - di_fixed[i]) ** 2 for i in range(len(s))
            )

        # Step 3: MM iterations with fixed d_i
        x_current = x_init.copy()
        for _ in range(max_iterations):
            x_prev = x_current.copy()

            numerator = np.zeros(2)
            denominator = 0
            for i in range(len(s)):
                direction = x_current - s[i]
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                numerator += s[i] + di_fixed[i] * (direction / norm_dir)
                denominator += 1
            x_current = numerator / denominator  # Eq. (26)

            if np.linalg.norm(x_current - x_prev) < tolerance:
                break

        lon, lat = self.utm_to_geo(x_current[0], x_current[1])
        return lat, lon


class SRLSLocalizer(CoordinateHandler):
    def __init__(self, hydrophones: list[Hydrophone], sound_speed=1500):
        super().__init__(hydrophones=hydrophones)
        self.v = sound_speed

    def _construct_matrix_A(self):
        s = self.get_utm_coords()
        m = len(s)

        A = []

        # compute matrix A as (17)
        for i in range(m):
            A.append([s[i][0] * (-2), s[i][1] * (-2), 1])

        A = np.array(A)

        return A

    def _construct_vector_b(self):
        # compute vector b as (17)
        coords = np.array(self.get_utm_coords())
        r_squared = np.array(
            [
                (hydro.observed_pressure[-1]["toa"] * self.v) ** 2
                for hydro in self.hydrophones
            ]
        )

        a_norm_squared = np.sum(coords**2, axis=1)

        return r_squared - a_norm_squared

    def _construct_matrix_D(self):
        # compute matrix D as (18)
        s = self.get_utm_coords()
        n = len(s[0])

        D = np.zeros((n + 1, n + 1))

        D[:n, :n] = np.eye(n)

        return D

    def _construct_vector_f(self):
        # compute vector f as (18)

        n = len(self.get_utm_coords()[0])
        f = np.zeros(n + 1)
        f[n] = -0.5

        return f

    def sr_ls_localize(self):
        # Compute matrix A and vector b as (17)
        A = self._construct_matrix_A()
        b = self._construct_vector_b()

        # Compute matrix D and vector f as (18)
        D = self._construct_matrix_D()
        f = self._construct_vector_f()

        # Find optimal lambda as (24)
        lambda_opt = self.find_optimal_lambda(A, b, D, f)

        # Compute position estimation
        source_position = self.calculate_position_estimate(lambda_opt, A, b, D, f)

        # return coord in (lat, lon) format
        lon, lat = self.utm_to_geo(source_position[0], source_position[1])
        return lat, lon

    def determine_lambda_interval(self, A, D):
        # Compute lower bound for interval I for lambda as eq. (26)

        # Compute A^T A
        ATA = A.T @ A

        # Compute lowest eigenvalue
        eigvals = scipy.linalg.eigvals(D, ATA)
        lambda_min = min(abs(eigvals))

        # Compute lower bound as (26)
        lower_bound = -1 / lambda_min + 1e-10

        return lower_bound

    def phi_function(self, lambda_val, A, b, D, f):
        # Compute value of φ(λ) as eq. (25)

        try:
            # Compute A^T A + λD
            ATA_plus_lambda_D = A.T @ A + lambda_val * D

            # Compute A^T b - λf
            ATb_minus_lambda_f = A.T @ b - lambda_val * f

            # Compute ŷ(λ) = (A^T A + λD)^(-1)(A^T b - λf)
            y_lambda = np.linalg.solve(ATA_plus_lambda_D, ATb_minus_lambda_f)

            # Compute φ(λ) = ŷ(λ)^T D ŷ(λ) + 2f^T ŷ(λ)
            phi_val = y_lambda.T @ D @ y_lambda + 2 * f.T @ y_lambda

            return phi_val
        except np.linalg.LinAlgError:
            return float("inf")

    def find_optimal_lambda(self, A, b, D, f):
        # Find optimal value of lambda as φ(λ) = 0, eq. (24)

        lower_bound = self.determine_lambda_interval(A, D)
        upper_bound = 1e6

        def phi_for_bisect(lambda_val):
            return self.phi_function(lambda_val, A, b, D, f)

        lower_val = phi_for_bisect(lower_bound)
        upper_val = phi_for_bisect(upper_bound)

        if lower_val * upper_val > 0:
            if abs(lower_val) < abs(upper_val):
                return lower_bound
            else:
                return upper_bound

        optimal_lambda = scipy.optimize.bisect(
            phi_for_bisect, lower_bound, upper_bound, rtol=1e-6
        )

        return optimal_lambda

    def calculate_position_estimate(self, lambda_opt, A, b, D, f):
        # Calculate pos estimation given optimal lambda as eq. (23)

        # Compute A^T A + λD
        ATA_plus_lambda_D = A.T @ A + lambda_opt * D

        # Compute A^T b - λf
        ATb_minus_lambda_f = A.T @ b - lambda_opt * f

        # Compute ŷ(λ) = (A^T A + λD)^(-1)(A^T b - λf)
        y_lambda = np.linalg.solve(ATA_plus_lambda_D, ATb_minus_lambda_f)

        n = len(self.get_utm_coords()[0])  # dimensione dello spazio
        source_position = y_lambda[:n]

        return source_position


class Core:
    @staticmethod
    def weighted_centroid_localization(hydrophones: list[Hydrophone]):
        localizer = WCLLocalizer(hydrophones)
        return localizer.weighted_centroid_localize()

    @staticmethod
    def tdoa_localization(hydrophones: list[Hydrophone]):
        localizer = TDOALocalizer(hydrophones)
        return localizer.tdoa_localize()

    @staticmethod
    def tmm_localization(hydrophones: list[Hydrophone]):
        localizer = TMMLocalizer(hydrophones)
        return localizer.tmm_localize()

    @staticmethod
    def tmm_localization_sr_ls_init(hydrophones: list[Hydrophone]):
        localizer = TMMLocalizer(hydrophones)
        return localizer.tmm_localize_sr_ls_init()

    @staticmethod
    def sr_ls_localization(hydrophones: list[Hydrophone]):
        localizer = SRLSLocalizer(hydrophones)
        return localizer.sr_ls_localize()
