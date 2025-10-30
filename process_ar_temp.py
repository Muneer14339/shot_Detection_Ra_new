import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CSVShotDetector:
    def __init__(self, fs=833):
        self.fs = fs
        self.dt = 1.0 / fs

        # Detection parameters
        self.peak_search_window = int(2.0 * fs)  # 2-second window for finding max peak
        self.analysis_half_window = 80           # 80 samples before/after -> total 161
        self.min_time_between_shots = 1.0        # seconds between shots
        self.pre_stability_window = (0.3, 0.15)  # 0.3–0.10 s before peak

        # Range criteria
        self.range_criteria = {
            "accel_peak": (2.5, 29),
            "rise_time": (2.4, 26),
            "fall_time": (2.4, 26),
            "duration": (5, 24),
        }

        # Cleaning parameters
        self.accel_threshold = 5.0  # g, for detecting potential outliers
        self.gyro_threshold = 50.0  # deg/s, minimum for valid high-accel event
        self.neighbor_threshold = 0.5  # fraction of accel_threshold for neighbors

        # Moving average filter parameter
        self.ma_window = 3  # Moving average window size (odd number, e.g., 5 samples)

    def clean_data_adaptive(self, accel, gyro):
        """
        Removes false acceleration spikes (no-motion bursts) but keeps real events
        that show correlated gyro activity or buildup before the peak.
        """
        cleaned = accel.copy()
        n = len(accel)
        baseline = np.median(accel)
        dev = np.abs(accel - baseline)
        accel_thr = 3.0  # g above baseline considered a potential spike
        gyro_thr = 30.0  # deg/s threshold for motion correlation
        max_spike_len = 20  # samples

        in_spike = False
        start = 0

        for i in range(n):
            if accel[i] - baseline > accel_thr and not in_spike:
                in_spike = True
                start = i
            elif in_spike and accel[i] - baseline < 0.5:
                end = i
                # Analyze region
                region_len = end - start
                region_gyro_mean = np.mean(gyro[start:end])
                pre_segment = accel[max(0, start - 5):start]
                buildup = np.mean(np.diff(pre_segment)) if len(pre_segment) > 3 else 0

                # Conditions for false spike
                is_false = (
                    region_len < max_spike_len and
                    region_gyro_mean < gyro_thr and
                    buildup < 0.05  # no rising trend before spike
                )

                if is_false:
                    fill_val = np.median(accel[max(0, start - 5):start])
                    cleaned[start:end] = fill_val
                    # print(f"Removed false spike at {start}-{end}: accel_mean={np.mean(accel[start:end]):.2f}g, gyro={region_gyro_mean:.1f}")
                in_spike = False

        return cleaned



    def moving_average_filter(self, data):
        """
        Apply a moving average filter to the input data with the specified window size.
        """
        return np.convolve(data, np.ones(self.ma_window) / self.ma_window, mode='same')

    def compute_rise_time_improved(self, s, baseline):
        """
        Computes rise time (10% to 90%) relative to the amplitude above the baseline.
        Input 's' should be the segment from the pre-stability window to the peak.
        """
        peak_idx = np.argmax(s)
        peak_val = s[peak_idx]

        # Calculate amplitude and thresholds relative to baseline
        amplitude = peak_val - baseline
        if amplitude <= 0:
            return 0.0

        ten_percent_level = baseline + 0.1 * amplitude
        ninety_percent_level = baseline + 0.9 * amplitude

        # Find start_idx (10% crossing)
        start_idx = 0
        for i in range(peak_idx, -1, -1):
            if s[i] < ten_percent_level:
                start_idx = i + 1
                break
        
        # Find end_idx (90% crossing)
        end_idx = peak_idx
        for i in range(start_idx, peak_idx + 1):
            if s[i] >= ninety_percent_level:
                end_idx = i
                break
        
        if end_idx < start_idx:
            end_idx = peak_idx
            
        return (end_idx - start_idx) * self.dt

    def compute_fall_time(self, s, baseline):
        peak_idx = np.argmax(s)
        peak_val = s[peak_idx]
        amplitude = peak_val - baseline
        ten = baseline + 0.1 * amplitude
        ninety = baseline + 0.9 * amplitude
        right = s[peak_idx:]
        fall_start = next((i for i, val in enumerate(right) if val < ninety), 0)
        fall_end = next((i for i, val in enumerate(right) if val < ten), len(right) - 1)
        return (fall_end - fall_start) * self.dt

    def compute_duration(self, s, baseline):
        peak_idx = np.argmax(s)
        peak_val = s[peak_idx]
        amplitude = peak_val - baseline
        half = baseline + 0.5 * amplitude
        left_half = next((i for i in range(peak_idx, -1, -1) if s[i] < half), peak_idx)
        right_half = next((i for i in range(peak_idx, len(s)) if s[i] < half), len(s) - 1)
        return (right_half - left_half) * self.dt

    def _check_ranges(self, cand_peak, cand_rise, cand_fall, cand_duration):
        low, high = self.range_criteria["accel_peak"]
        if not (low <= cand_peak <= high):
            print(f"Candidate shot rejected: accel_peak ({cand_peak:.2f}g) outside range [{low}, {high}]")
            return False
        
        low, high = self.range_criteria["rise_time"]
        rise_ms = cand_rise * 1000
        if not (low <= rise_ms <= high):
            print(f"Candidate shot rejected: rise_time ({rise_ms:.2f}ms) outside range [{low}, {high}]")
            return False
        
        low, high = self.range_criteria["fall_time"]
        fall_ms = cand_fall * 1000
        if not (low <= fall_ms <= high):
            print(f"Candidate shot rejected: fall_time ({fall_ms:.2f}ms) outside range [{low}, {high}]")
            return False
        
        low, high = self.range_criteria["duration"]
        dur_ms = cand_duration * 1000
        if not (low <= dur_ms <= high):
            print(f"Candidate shot rejected: duration ({dur_ms:.2f}ms) outside range [{low}, {high}]")
            return False
        
        return True

    def detect_shots(self, accel, gyro):
        """
        Detect shots in cleaned and smoothed acceleration data, with detailed
        rejection reason logs for transparency.
        """
        cleaned_accel = self.clean_data_adaptive(accel, gyro)
        smoothed_accel = self.moving_average_filter(cleaned_accel)
        min_height = self.range_criteria["accel_peak"][0]
        detected = []
        n = len(smoothed_accel)
        i = 0

        print("\n--- Shot Detection Log ---")

        while i < n - self.peak_search_window:
            search_start = i
            search_end = min(i + self.peak_search_window, n)
            search_window = smoothed_accel[search_start:search_end]

            peak_rel = np.argmax(search_window)
            peak_idx = search_start + peak_rel
            peak_val = smoothed_accel[peak_idx]
            ts = peak_idx / self.fs  # Timestamp in seconds

            # --- 1️⃣ Minimum height check ---
            if peak_val < min_height:
                print(f"[{ts:.3f}s] ❌ Rejected candidate: Peak too low ({peak_val:.2f}g < {min_height}g)")
                i += int(0.5 * self.fs)
                continue

            # --- 2️⃣ Pre-trigger stability check ---
            pre_start = max(0, peak_idx - int(self.pre_stability_window[0] * self.fs))
            pre_end = max(0, peak_idx - int(self.pre_stability_window[1] * self.fs))
            pre_window = smoothed_accel[pre_start:pre_end]

            if len(pre_window) == 0:
                print(f"[{ts:.3f}s] ⚠️ Skipped: No pre-window samples available.")
                i += int(0.5 * self.fs)
                continue

            baseline = np.mean(pre_window)
            deviation = pre_window - baseline
            pre_range = np.max(np.abs(deviation))
            pre_std = np.std(deviation)
            formatted_samples = ", ".join(f"{v:.4f}" for v in pre_window)
            # print(
            #     f"\n[{ts:.3f}s] 📊 Pre-shot stability data ({len(pre_window)} samples):"
            #     f"\n  Raw: [{formatted_samples}]"
            #     f"\n  Baseline: {baseline:.4f} g"
            #     f"\n  Range: {pre_range:.4f} g"
            #     f"\n  Std: {pre_std:.4f} g"
            # )

            if len(pre_window) > 5 and (pre_range > 0.4 or pre_std > 0.15):
                print(f"[{ts:.3f}s] ❌ Rejected candidate: Unstable pre-shot motion "
                    f"(range={pre_range:.3f}g, std={pre_std:.3f}g)")
                i = peak_idx + int(0.5 * self.fs)
                continue

            # --- 3️⃣ Compute window-based metrics ---
            win_start = max(0, peak_idx - self.analysis_half_window)
            win_end = min(n, peak_idx + self.analysis_half_window + 1)
            local_window = smoothed_accel[win_start:win_end]

            cand_peak = np.max(local_window)
            rise_segment = smoothed_accel[pre_end:peak_idx + 1]
            cand_rise = self.compute_rise_time_improved(rise_segment, baseline)
            cand_fall = self.compute_fall_time(local_window, baseline)
            cand_dur = self.compute_duration(local_window, baseline)

            # --- 4️⃣ Range-based validation ---
            if self._check_ranges_verbose(cand_peak, cand_rise, cand_fall, cand_dur, ts):
                detected.append({
                    "index": peak_idx,
                    "time_s": ts,
                    "accel_peak": cand_peak,
                    "rise_time_ms": cand_rise * 1000,
                    "fall_time_ms": cand_fall * 1000,
                    "duration_ms": cand_dur * 1000
                })
                print(f"[{ts:.3f}s] ✅ Shot detected! Peak={cand_peak:.2f}g")
                i = peak_idx + int(self.min_time_between_shots * self.fs)
            else:
                i += int(0.25 * self.fs)

        return detected, cleaned_accel, smoothed_accel


    def _check_ranges_verbose(self, cand_peak, cand_rise, cand_fall, cand_dur, ts):
        """
        Like _check_ranges(), but prints explicit reasons for rejection with timestamp.
        """
        low, high = self.range_criteria["accel_peak"]
        if not (low <= cand_peak <= high):
            print(f"[{ts:.3f}s] ❌ Rejected: accel_peak={cand_peak:.2f}g (expected {low}-{high})")
            return False

        low, high = self.range_criteria["rise_time"]
        rise_ms = cand_rise * 1000
        if not (low <= rise_ms <= high):
            print(f"[{ts:.3f}s] ❌ Rejected: rise_time={rise_ms:.2f}ms (expected {low}-{high})")
            return False

        low, high = self.range_criteria["fall_time"]
        fall_ms = cand_fall * 1000
        if not (low <= fall_ms <= high):
            print(f"[{ts:.3f}s] ❌ Rejected: fall_time={fall_ms:.2f}ms (expected {low}-{high})")
            return False

        low, high = self.range_criteria["duration"]
        dur_ms = cand_dur * 1000
        if not (low <= dur_ms <= high):
            print(f"[{ts:.3f}s] ❌ Rejected: duration={dur_ms:.2f}ms (expected {low}-{high})")
            return False

        return True

  
def main():
    filename = "new\live_session_down.csv"
    df = pd.read_csv(filename)

    required_columns = ["Acceleration Magnitude", "Gyro Magnitude"]
    if not all(col in df.columns for col in required_columns):
        print(f"CSV must have {', '.join(required_columns)} columns.")
        return

    accel = df["Acceleration Magnitude"].to_numpy()
    gyro = df["Gyro Magnitude"].to_numpy()
    fs = int(1.0 / 0.0012)  # 1.2 ms per sample → ~833 Hz

    detector = CSVShotDetector(fs=fs)
    shots, cleaned_accel, smoothed_accel = detector.detect_shots(accel, gyro)

    print(f"\nTotal shots detected: {len(shots)}")

    if shots:
        print("\nShot Summary:")
        print(f"{'No.':<4} {'Time(s)':<10} {'Peak(g)':<10} {'Rise(ms)':<10} {'Fall(ms)':<10} {'Dur(ms)':<10}")
        print("-" * 58)
        for n, s in enumerate(shots, 1):
            print(f"{n:<4} {s['time_s']:<10.3f} {s['accel_peak']:<10.2f} {s['rise_time_ms']:<10.2f} "
                  f"{s['fall_time_ms']:<10.2f} {s['duration_ms']:<10.2f}")

    # --- Plot results ---
    plt.figure(figsize=(12, 6))
    plt.plot(accel, label="Original Acceleration", color="blue", alpha=0.5)
    plt.plot(cleaned_accel, label="Cleaned Acceleration", color="green", alpha=0.7)
    plt.plot(smoothed_accel, label="Smoothed Acceleration", color="purple", linewidth=2)
    if shots:
        shot_indices = [s["index"] for s in shots]
        plt.scatter(shot_indices, smoothed_accel[shot_indices], color="red", label="Detected Shots", zorder=3)
    plt.title("Shot Detection with Cleaned and Smoothed Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration Magnitude (g)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()






# both up and down 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# class CSVShotDetector:
#     def __init__(self, fs=833):
#         self.fs = fs
#         self.dt = 1.0 / fs

#         # Detection parameters
#         self.peak_search_window = int(2.0 * fs)  # 2-second window for finding max peak
#         self.analysis_half_window = 80           # 80 samples before/after -> total 161
#         self.min_time_between_shots = 1.0        # seconds between shots
#         self.pre_stability_window = (0.3, 0.10)  # 0.3–0.15 s before peak

#         # Range criteria
#         self.range_criteria = {
#             "accel_peak": (3.5, 20),
#             "rise_time": (2.4, 17),
#             "fall_time": (2.4, 26),
#             "duration": (2.4, 17),
#         }

#         # Cleaning parameters
#         self.accel_threshold = 5.0  # g, for detecting potential outliers
#         self.gyro_threshold = 50.0  # deg/s, minimum for valid high-accel event
#         self.neighbor_threshold = 0.5  # fraction of accel_threshold for neighbors

#         # Moving average filter parameter
#         self.ma_window = 3  # Moving average window size (odd number, e.g., 5 samples)

#     def clean_data_adaptive(self, accel, gyro):
#         """
#         Removes false acceleration spikes (no-motion bursts) but keeps real events
#         that show correlated gyro activity or buildup before the peak.
#         """
#         cleaned = accel.copy()
#         n = len(accel)
#         baseline = np.median(accel)
#         dev = np.abs(accel - baseline)
#         accel_thr = 3.0  # g above baseline considered a potential spike
#         gyro_thr = 30.0  # deg/s threshold for motion correlation
#         max_spike_len = 20  # samples

#         in_spike = False
#         start = 0

#         for i in range(n):
#             if accel[i] - baseline > accel_thr and not in_spike:
#                 in_spike = True
#                 start = i
#             elif in_spike and accel[i] - baseline < 0.5:
#                 end = i
#                 # Analyze region
#                 region_len = end - start
#                 region_gyro_mean = np.mean(gyro[start:end])
#                 pre_segment = accel[max(0, start - 5):start]
#                 buildup = np.mean(np.diff(pre_segment)) if len(pre_segment) > 3 else 0

#                 # Conditions for false spike
#                 is_false = (
#                     region_len < max_spike_len and
#                     region_gyro_mean < gyro_thr and
#                     buildup < 0.05  # no rising trend before spike
#                 )

#                 if is_false:
#                     fill_val = np.median(accel[max(0, start - 5):start])
#                     cleaned[start:end] = fill_val
#                     # print(f"Removed false spike at {start}-{end}: accel_mean={np.mean(accel[start:end]):.2f}g, gyro={region_gyro_mean:.1f}")
#                 in_spike = False

#         return cleaned



#     def moving_average_filter(self, data):
#         """
#         Apply a moving average filter to the input data with the specified window size.
#         """
#         return np.convolve(data, np.ones(self.ma_window) / self.ma_window, mode='same')

#     def compute_rise_time_improved(self, s, baseline):
#         """
#         Computes rise time (10% to 90%) relative to the amplitude above the baseline.
#         Input 's' should be the segment from the pre-stability window to the peak.
#         """
#         peak_idx = np.argmax(s)
#         peak_val = s[peak_idx]

#         # Calculate amplitude and thresholds relative to baseline
#         amplitude = peak_val - baseline
#         if amplitude <= 0:
#             return 0.0

#         ten_percent_level = baseline + 0.1 * amplitude
#         ninety_percent_level = baseline + 0.9 * amplitude

#         # Find start_idx (10% crossing)
#         start_idx = 0
#         for i in range(peak_idx, -1, -1):
#             if s[i] < ten_percent_level:
#                 start_idx = i + 1
#                 break
        
#         # Find end_idx (90% crossing)
#         end_idx = peak_idx
#         for i in range(start_idx, peak_idx + 1):
#             if s[i] >= ninety_percent_level:
#                 end_idx = i
#                 break
        
#         if end_idx < start_idx:
#             end_idx = peak_idx
            
#         return (end_idx - start_idx) * self.dt

#     def compute_fall_time(self, s, baseline):
#         peak_idx = np.argmax(s)
#         peak_val = s[peak_idx]
#         amplitude = peak_val - baseline
#         ten = baseline + 0.1 * amplitude
#         ninety = baseline + 0.9 * amplitude
#         right = s[peak_idx:]
#         fall_start = next((i for i, val in enumerate(right) if val < ninety), 0)
#         fall_end = next((i for i, val in enumerate(right) if val < ten), len(right) - 1)
#         return (fall_end - fall_start) * self.dt

#     def compute_duration(self, s, baseline):
#         peak_idx = np.argmax(s)
#         peak_val = s[peak_idx]
#         amplitude = peak_val - baseline
#         half = baseline + 0.5 * amplitude
#         left_half = next((i for i in range(peak_idx, -1, -1) if s[i] < half), peak_idx)
#         right_half = next((i for i in range(peak_idx, len(s)) if s[i] < half), len(s) - 1)
#         return (right_half - left_half) * self.dt

#     def _check_ranges(self, cand_peak, cand_rise, cand_fall, cand_duration):
#         low, high = self.range_criteria["accel_peak"]
#         if not (low <= cand_peak <= high):
#             print(f"Candidate shot rejected: accel_peak ({cand_peak:.2f}g) outside range [{low}, {high}]")
#             return False
        
#         low, high = self.range_criteria["rise_time"]
#         rise_ms = cand_rise * 1000
#         if not (low <= rise_ms <= high):
#             print(f"Candidate shot rejected: rise_time ({rise_ms:.2f}ms) outside range [{low}, {high}]")
#             return False
        
#         low, high = self.range_criteria["fall_time"]
#         fall_ms = cand_fall * 1000
#         if not (low <= fall_ms <= high):
#             print(f"Candidate shot rejected: fall_time ({fall_ms:.2f}ms) outside range [{low}, {high}]")
#             return False
        
#         low, high = self.range_criteria["duration"]
#         dur_ms = cand_duration * 1000
#         if not (low <= dur_ms <= high):
#             print(f"Candidate shot rejected: duration ({dur_ms:.2f}ms) outside range [{low}, {high}]")
#             return False
        
#         return True

#     def detect_shots(self, accel, gyro):
#         """
#         Detect shots in cleaned and smoothed acceleration data, with detailed
#         rejection reason logs for transparency.
#         """
#         cleaned_accel = self.clean_data_adaptive(accel, gyro)
#         smoothed_accel = self.moving_average_filter(cleaned_accel)
#         min_height = self.range_criteria["accel_peak"][0]
#         detected = []
#         n = len(smoothed_accel)
#         i = 0

#         print("\n--- Shot Detection Log ---")

#         while i < n - self.peak_search_window:
#             search_start = i
#             search_end = min(i + self.peak_search_window, n)
#             search_window = smoothed_accel[search_start:search_end]

#             peak_rel = np.argmax(search_window)
#             peak_idx = search_start + peak_rel
#             peak_val = smoothed_accel[peak_idx]
#             ts = peak_idx / self.fs  # Timestamp in seconds

#             # --- 1️⃣ Minimum height check ---
#             if peak_val < min_height:
#                 print(f"[{ts:.3f}s] ❌ Rejected candidate: Peak too low ({peak_val:.2f}g < {min_height}g)")
#                 i += int(0.5 * self.fs)
#                 continue

#             # --- 2️⃣ Pre-trigger stability check ---
#             pre_start = max(0, peak_idx - int(self.pre_stability_window[0] * self.fs))
#             pre_end = max(0, peak_idx - int(self.pre_stability_window[1] * self.fs))
#             pre_window = smoothed_accel[pre_start:pre_end]

#             if len(pre_window) == 0:
#                 print(f"[{ts:.3f}s] ⚠️ Skipped: No pre-window samples available.")
#                 i += int(0.5 * self.fs)
#                 continue

#             baseline = np.mean(pre_window)
#             deviation = pre_window - baseline
#             pre_range = np.max(np.abs(deviation))
#             pre_std = np.std(deviation)

#             if len(pre_window) > 5 and (pre_range > 0.15 or pre_std > 0.05):
#                 print(f"[{ts:.3f}s] ❌ Rejected candidate: Unstable pre-shot motion "
#                     f"(range={pre_range:.3f}g, std={pre_std:.3f}g)")
#                 i = peak_idx + int(0.5 * self.fs)
#                 continue

#             # --- 3️⃣ Compute window-based metrics ---
#             win_start = max(0, peak_idx - self.analysis_half_window)
#             win_end = min(n, peak_idx + self.analysis_half_window + 1)
#             local_window = smoothed_accel[win_start:win_end]

#             cand_peak = np.max(local_window)
#             rise_segment = smoothed_accel[pre_end:peak_idx + 1]
#             cand_rise = self.compute_rise_time_improved(rise_segment, baseline)
#             cand_fall = self.compute_fall_time(local_window, baseline)
#             cand_dur = self.compute_duration(local_window, baseline)

#             # --- 4️⃣ Range-based validation ---
#             if self._check_ranges_verbose(cand_peak, cand_rise, cand_fall, cand_dur, ts):
#                 detected.append({
#                     "index": peak_idx,
#                     "time_s": ts,
#                     "accel_peak": cand_peak,
#                     "rise_time_ms": cand_rise * 1000,
#                     "fall_time_ms": cand_fall * 1000,
#                     "duration_ms": cand_dur * 1000
#                 })
#                 print(f"[{ts:.3f}s] ✅ Shot detected! Peak={cand_peak:.2f}g")
#                 i = peak_idx + int(self.min_time_between_shots * self.fs)
#             else:
#                 i += int(0.25 * self.fs)

#         return detected, cleaned_accel, smoothed_accel


#     def _check_ranges_verbose(self, cand_peak, cand_rise, cand_fall, cand_dur, ts):
#         """
#         Like _check_ranges(), but prints explicit reasons for rejection with timestamp.
#         """
#         low, high = self.range_criteria["accel_peak"]
#         if not (low <= cand_peak <= high):
#             print(f"[{ts:.3f}s] ❌ Rejected: accel_peak={cand_peak:.2f}g (expected {low}-{high})")
#             return False

#         low, high = self.range_criteria["rise_time"]
#         rise_ms = cand_rise * 1000
#         if not (low <= rise_ms <= high):
#             print(f"[{ts:.3f}s] ❌ Rejected: rise_time={rise_ms:.2f}ms (expected {low}-{high})")
#             return False

#         low, high = self.range_criteria["fall_time"]
#         fall_ms = cand_fall * 1000
#         if not (low <= fall_ms <= high):
#             print(f"[{ts:.3f}s] ❌ Rejected: fall_time={fall_ms:.2f}ms (expected {low}-{high})")
#             return False

#         low, high = self.range_criteria["duration"]
#         dur_ms = cand_dur * 1000
#         if not (low <= dur_ms <= high):
#             print(f"[{ts:.3f}s] ❌ Rejected: duration={dur_ms:.2f}ms (expected {low}-{high})")
#             return False

#         return True


# def main():
#     filename = "live_sk_ud.csv"
#     df = pd.read_csv(filename)

#     required_columns = ["Acceleration Magnitude", "Gyro Magnitude"]
#     if not all(col in df.columns for col in required_columns):
#         print(f"CSV must have {', '.join(required_columns)} columns.")
#         return

#     accel = df["Acceleration Magnitude"].to_numpy()
#     gyro = df["Gyro Magnitude"].to_numpy()
#     fs = int(1.0 / 0.0012)  # 1.2 ms per sample → ~833 Hz

#     detector = CSVShotDetector(fs=fs)
#     shots, cleaned_accel, smoothed_accel = detector.detect_shots(accel, gyro)

#     print(f"\nTotal shots detected: {len(shots)}")

#     if shots:
#         print("\nShot Summary:")
#         print(f"{'No.':<4} {'Time(s)':<10} {'Peak(g)':<10} {'Rise(ms)':<10} {'Fall(ms)':<10} {'Dur(ms)':<10}")
#         print("-" * 58)
#         for n, s in enumerate(shots, 1):
#             print(f"{n:<4} {s['time_s']:<10.3f} {s['accel_peak']:<10.2f} {s['rise_time_ms']:<10.2f} "
#                   f"{s['fall_time_ms']:<10.2f} {s['duration_ms']:<10.2f}")

#     # --- Plot results ---
#     plt.figure(figsize=(12, 6))
#     plt.plot(accel, label="Original Acceleration", color="blue", alpha=0.5)
#     plt.plot(cleaned_accel, label="Cleaned Acceleration", color="green", alpha=0.7)
#     plt.plot(smoothed_accel, label="Smoothed Acceleration", color="purple", linewidth=2)
#     if shots:
#         shot_indices = [s["index"] for s in shots]
#         plt.scatter(shot_indices, smoothed_accel[shot_indices], color="red", label="Detected Shots", zorder=3)
#     plt.title("Shot Detection with Cleaned and Smoothed Data")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Acceleration Magnitude (g)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     main()

