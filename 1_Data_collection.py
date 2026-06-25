import time
import threading
import pandas as pd
import os
from pyomyo import Myo, emg_mode


def data_worker(mode, stop_event, filepath):
    m = None
    myo_data = []

    try:
        print("[WORKER] Connecting to the Myo...")
        m = Myo(mode=mode)
        m.disconnect()
        m.connect()
        print("[WORKER] Connected to the Myo.")


        # GREEN LED: connected
        try:
            m.set_leds([0, 128, 0], [0, 128, 0])
            print("Device connected! Green LED indicates connection.")
        except Exception as e:
            print(f"[WORKER] Unable to set the green LED: {e}")

        # EMG handler
        def add_to_queue(emg, movement):
            myo_data.append(emg)
            if len(myo_data) % 50 == 0:
                print(f"[WORKER] Collected {len(myo_data)} samples so far...")

        m.add_emg_handler(add_to_queue)

        # Battery handler 
        def print_battery(bat):
            print("Battery level:", bat)

        m.add_battery_handler(print_battery)

        # Initial vibration
        try:
            m.vibrate(2)
            print("[WORKER] Device vibrating... get ready!")
        except Exception as e:
            print(f"[WORKER] Unable to make the Myo vibrate: {e}")

        # time.sleep(2)

        # BLUE LED: start of collection
        try:
            m.set_leds([0, 0, 128], [0, 0, 128])
            print("[WORKER] Data collection started (blue LED).")
        except Exception as e:
            print(f"[WORKER] Unable to set the blue LED: {e}")

        # Main loop
        while not stop_event.is_set():
            m.run()
            time.sleep(0.01)

            # Periodic save every 100 samples
            if len(myo_data) > 0 and len(myo_data) % 100 == 0:
                try:
                    myo_cols = [f"Channel_{i}" for i in range(1, 9)]
                    myo_df = pd.DataFrame(myo_data, columns=myo_cols)
                    myo_df.to_csv(filepath, index=False)
                    print(f"[WORKER] Periodic save, samples: {len(myo_data)}")
                    print(f"[WORKER] Periodic file at: {os.path.abspath(filepath)}")
                except Exception as e:
                    print(f"[WORKER] Error during periodic save: {e}")

        print("[WORKER] stop_event active, exiting the loop...")

    except Exception as e:
        print(f"[WORKER] Error during data collection: {e}")

    finally:
        # Always save a file, even if it is empty
        try:
            myo_cols = [f"Channel_{i}" for i in range(1, 9)]
            myo_df = pd.DataFrame(myo_data, columns=myo_cols)
            myo_df.to_csv(filepath, index=False)
            print(f"[WORKER] Final CSV saved at: {os.path.abspath(filepath)}")
            print(f"[WORKER] Total samples saved: {len(myo_data)}")
        except Exception as e:
            print(f"[WORKER] Error while saving final CSV: {e}")

        # YELLOW LED + disconnect
        if m is not None:
            try:
                m.set_leds([128, 128, 0], [128, 128, 0])
                print("[WORKER] Yellow LED (end).")
                time.sleep(1)
            except Exception:
                print("[WORKER] Unable to set the yellow LED.")

            try:
                m.disconnect()
                print("[WORKER] Myo disconnected.")
            except Exception:
                print("[WORKER] Unable to disconnect the Myo.")


# -------- Main Program Loop -----------
if __name__ == '__main__':
    stop_event = threading.Event()

    # === GENERIC OUTPUT PATH ===
    base_dir = r"PATH_TO_YOUR_OUTPUT_FOLDER"
    os.makedirs(base_dir, exist_ok=True)  # create the folder if it does not exist
    print("[MAIN] I will save the data in:", base_dir)

    # Output file name
    file_name = os.path.join(base_dir, "participant_data.csv")
    print("[MAIN] File name:", file_name)
    # =======================

    mode = emg_mode.PREPROCESSED

    # Avvio THREAD invece di PROCESSO
    worker_thread = threading.Thread(
        target=data_worker,
        args=(mode, stop_event, file_name),
        daemon=True
    )
    worker_thread.start()

    print("Press ENTER to stop data collection...")

    try:
        input()  #Enter for a clean stop
        print("[MAIN] Enter pressed, stopping the worker...")
        stop_event.set()
        worker_thread.join()
        print("[MAIN] Data collection finished.")

    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C detected, stopping the worker...")
        stop_event.set()
        worker_thread.join()
        print("[MAIN] Data collection finished (Ctrl+C).")
