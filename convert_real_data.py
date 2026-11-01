import os
import re
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

ACTIVITY_MAP = {
    "Standing": "standing",
    "Walking": "walking",
    "Jumping": "jumping",
    "Still": "still",
}

N_TRAIN_PER_ACTIVITY = 3 
N_TEST_PER_ACTIVITY = 2    


def parse_session_folder(folder_name):
    """
    Extract activity key from folder name like 'Walking_1-2026-11-01_12-16-06'.
    Returns activity key (e.g. 'Walking') or None.
    """
    m = re.match(r"^(Standing|Walking|Jumping|Still)_\d+-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", folder_name)
    if m:
        return m.group(1)
    return None


def compute_fs(seconds_elapsed):
    """Estimate sampling rate (Hz) from median time difference."""
    if len(seconds_elapsed) < 2:
        return 50.0
    diffs = np.diff(seconds_elapsed)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 50.0
    median_dt = np.median(diffs)
    if median_dt <= 0:
        return 50.0
    return 1.0 / median_dt


def read_total_accel(filepath):
    """
    Read TotalAcceleration.csv (time, seconds_elapsed, z, y, x).
    Returns timestamps (seconds_elapsed), accel_x, accel_y, accel_z, fs.
    """
    timestamps = []
    z_vals = []
    y_vals = []
    x_vals = []

    with open(filepath, "r") as f:
        header = f.readline() 
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            timestamps.append(float(parts[1]))  
            z_vals.append(float(parts[2]))       
            y_vals.append(float(parts[3]))       
            x_vals.append(float(parts[4]))       

    timestamps = np.array(timestamps)
    
    accel_x = np.array(x_vals)
    accel_y = np.array(y_vals)
    accel_z = np.array(z_vals)
    fs = compute_fs(timestamps)
    return timestamps, accel_x, accel_y, accel_z, fs


def write_pipeline_csv(filepath, timestamps, accel_x, accel_y, accel_z, fs, activity):
    """Write a CSV in the format the pipeline expects."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    n = len(timestamps)
    gyro = np.zeros(n)
    with open(filepath, "w") as f:
        f.write(f"# sampling_rate_hz={fs:.2f}\n")
        f.write(f"# activity={activity}\n")
        f.write("timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
        for i in range(n):
            f.write(f"{timestamps[i]:.6f},{accel_x[i]:.6f},{accel_y[i]:.6f},{accel_z[i]:.6f},"
                    f"{gyro[i]:.6f},{gyro[i]:.6f},{gyro[i]:.6f}\n")


def main():
   
    sessions_by_activity = {a: [] for a in ACTIVITY_MAP.values()}

    for entry in sorted(os.listdir(DATA_DIR)):
        full_path = os.path.join(DATA_DIR, entry)
        if not os.path.isdir(full_path):
            continue
        activity_key = parse_session_folder(entry)
        if activity_key is None:
            continue
        activity = ACTIVITY_MAP[activity_key]
        sessions_by_activity[activity].append(full_path)

    print("Found sessions per activity:")
    for activity, sessions in sessions_by_activity.items():
        print(f"  {activity}: {len(sessions)} sessions")

    
    for activity in sessions_by_activity:
        os.makedirs(os.path.join(TRAIN_DIR, activity), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, activity), exist_ok=True)

    total_train = 0
    total_test = 0

    for activity, sessions in sessions_by_activity.items():
        
        sessions.sort()
        train_sessions = sessions[:N_TRAIN_PER_ACTIVITY]
        test_sessions = sessions[N_TRAIN_PER_ACTIVITY:N_TRAIN_PER_ACTIVITY + N_TEST_PER_ACTIVITY]

        
        for i, sess_dir in enumerate(train_sessions, start=1):
            accel_file = os.path.join(sess_dir, "TotalAcceleration.csv")
            if not os.path.exists(accel_file):
                print(f"  SKIP (no TotalAcceleration.csv): {sess_dir}")
                continue
            timestamps, ax, ay, az, fs = read_total_accel(accel_file)
            fname = f"{activity}_train_{i:02d}.csv"
            out_path = os.path.join(TRAIN_DIR, activity, fname)
            write_pipeline_csv(out_path, timestamps, ax, ay, az, fs, activity)
            total_train += 1
            print(f"  TRAIN {fname}: {len(timestamps)} samples, fs={fs:.1f} Hz, "
                  f"duration={timestamps[-1]:.1f}s")

    
        for i, sess_dir in enumerate(test_sessions, start=1):
            accel_file = os.path.join(sess_dir, "TotalAcceleration.csv")
            if not os.path.exists(accel_file):
                print(f"  SKIP (no TotalAcceleration.csv): {sess_dir}")
                continue
            timestamps, ax, ay, az, fs = read_total_accel(accel_file)
            fname = f"{activity}_test_{i:02d}.csv"
            out_path = os.path.join(TEST_DIR, activity, fname)
            write_pipeline_csv(out_path, timestamps, ax, ay, az, fs, activity)
            total_test += 1
            print(f"  TEST  {fname}: {len(timestamps)} samples, fs={fs:.1f} Hz, "
                  f"duration={timestamps[-1]:.1f}s")

    print(f"\nDone. Train: {total_train} files, Test: {total_test} files")


if __name__ == "__main__":
    main()
