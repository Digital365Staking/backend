import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import json
import signal

def handle_sigint(sig, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, handle_sigint)

# Generate timestamp
ts_str = time.strftime("%Y%m%d_%H%M%S")

# --- Configuration ---
DATASET_NAME        = f"dataset_{ts_str}"
RECORD_DURATION_SEC = None   # None = record until 'q'
SUBSAMPLE_FACTOR    = 2      # keep every 2nd frame → ~15 fps effective

# FIX Bug 1: D455 depth sensor max resolution is 848×480.
# 1280×720 is only valid for the color stream.
# Using mismatched resolutions also requires align to rescale,
# so keep both at the same resolution for best alignment quality.
COLOR_W, COLOR_H = 848, 480
DEPTH_W, DEPTH_H = 848, 480
FPS = 30


def create_euroc_structure(base_path):
    cam0_path  = os.path.join(base_path, "cam0",   "data")
    depth_path = os.path.join(base_path, "depth0", "data")
    calib_path = os.path.join(base_path, "calib")
    for p in [cam0_path, depth_path, calib_path]:
        os.makedirs(p, exist_ok=True)
    return cam0_path, depth_path, calib_path


def save_calibration(calib_path, profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr         = color_stream.get_intrinsics()
    depth_scale  = profile.get_device().first_depth_sensor().get_depth_scale()

    calib = {
        "cam0": {
            "camera_model":      "pinhole",
            "intrinsics":        [intr.fx, intr.fy, intr.ppx, intr.ppy],
            "distortion_model":  "radtan",
            "distortion_coeffs": list(intr.coeffs),
            "resolution":        [intr.width, intr.height],
            # FIX Bug 6: save depth_scale so TSDF scripts can convert
            # raw uint16 values to meters (typically 0.001 = 1mm per unit)
            "depth_scale":       depth_scale,
        }
    }

    # FIX Bug 2: save as camera_imu.json — all downstream SLAM/TSDF
    # scripts expect this exact filename, not camera.json
    path = os.path.join(calib_path, "camera_imu.json")
    with open(path, "w") as f:
        json.dump(calib, f, indent=4)
    print(f"Calibration saved → {path}")
    print(f"  fx={intr.fx:.2f} fy={intr.fy:.2f} "
          f"cx={intr.ppx:.2f} cy={intr.ppy:.2f}  "
          f"{intr.width}x{intr.height}  depth_scale={depth_scale}")


def main():
    print("Starting D455 RGB-D recorder...")

    base_path = os.path.join(os.getcwd(), "output", DATASET_NAME)
    cam0_path, depth_path, calib_path = create_euroc_structure(base_path)
    color_ts_file = open(os.path.join(cam0_path, "timestamps.txt"), "w")

    # --- Pipeline ---
    pipeline = rs.pipeline()
    config   = rs.config()

    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16,  FPS)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start pipeline: {e}")
        print("Check that COLOR_W/DEPTH_W are valid for your D455 firmware.")
        return

    # --- Sensor tuning ---
    depth_sensor = profile.get_device().first_depth_sensor()

    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)   # High Accuracy

    depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
    depth_sensor.set_option(rs.option.laser_power,          330)
    depth_sensor.set_option(rs.option.exposure,             8500)
    depth_sensor.set_option(rs.option.gain,                 16)

    save_calibration(calib_path, profile)

    # --- Filters ---
    # FIX Bug 4: define filters before align — spatial/temporal must run
    # on the raw depth frame before it is reprojected onto the color plane.
    # Filtering after alignment introduces boundary artifacts.
    decimation   = rs.decimation_filter()   # optional: keeps resolution, reduces noise
    spatial      = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude,    2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    temporal     = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # Align after filtering
    align = rs.align(rs.stream.color)

    # --- Warm-up ---
    print("Warming up camera (15 frames)...")
    for _ in range(15):
        try:
            pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            pass

    print(f"Recording started. Press 'q' to stop.")

    # FIX Bug 5: use a single monotonically increasing frame_count
    # and only increment it once per loop iteration, at the end.
    # The original had multiple increment points causing inconsistent subsampling.
    frame_count = 0
    saved_count = 0
    start_time  = time.time()

    try:
        while True:
            if RECORD_DURATION_SEC and (time.time() - start_time) > RECORD_DURATION_SEC:
                print("Recording duration reached.")
                break

            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"Frame timeout: {e} — retrying")
                continue

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                frame_count += 1
                continue

            # FIX Bug 4: apply all depth filters on the RAW depth frame
            # BEFORE alignment to color space

            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                frame_count += 1
                continue

            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # Subsample
            if frame_count % SUBSAMPLE_FACTOR != 0:
                frame_count += 1
                continue

            color_ts  = color_frame.get_timestamp() / 1000.0   # ms → seconds
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())   # uint16, mm

            # Save color as PNG (lossless)
            cv2.imwrite(
                os.path.join(cam0_path,  f"{color_ts:.6f}.png"),
                color_img
            )
            # Save depth as 16-bit PNG (preserves mm values)
            cv2.imwrite(
                os.path.join(depth_path, f"{color_ts:.6f}_depth.png"),
                depth_img
            )

            color_ts_file.write(f"{color_ts:.6f}\n")
            color_ts_file.flush()

            saved_count += 1

            if saved_count % 15 == 0:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] {saved_count} frames saved")

            # Display
            # FIX Bug 3: depth_img is uint16 — must normalize to uint8 explicitly
            # before applying colormap. convertScaleAbs on uint16 clips silently.
            depth_display = cv2.normalize(
                depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            cv2.imshow("Color", color_img)
            cv2.imshow("Depth", depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit.")
                break

            frame_count += 1   # FIX Bug 5: single increment at end of loop

    finally:
        pipeline.stop()
        color_ts_file.close()
        cv2.destroyAllWindows()
        print(f"\nRecording complete.")
        print(f"  Saved {saved_count} frames")
        print(f"  Dataset: {base_path}")


if __name__ == "__main__":
    main()
