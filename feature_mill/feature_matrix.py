import os
import sqlite3
import pandas as pd
import gc
import argparse
import time
import psutil
from multiprocessing import Pool, cpu_count, TimeoutError
from eeg2fx.featureset_fetcher import run_feature_set
from eeg2fx.featureset_grouping import load_fxdefs_for_set

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_recording_ids_for_dataset(dataset_id: str) -> list[int]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM recordings WHERE dataset_id = ?", (dataset_id,))
    ids = [row[0] for row in c.fetchall()]
    conn.close()
    return ids

def get_fxdef_meta(fxid: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT shortname, chans FROM fxdef WHERE id = ?", (fxid,))
    row = c.fetchone()
    conn.close()
    return {
        "shortname": row[0] if row else f"fx{fxid}",
        "chans": row[1] if row else "NA"
    }

def process_one_recording(args):
    recording_id, feature_set_id, output_dir = args
    try:
        print(f"[matrix]Processing recording {recording_id} with feature set {feature_set_id}")
        fx_values = run_feature_set(feature_set_id, recording_id)
        scalar_row = {"recording_id": recording_id}
        epoch_rows = []

        for fxid, fxval in fx_values.items():
            fxmeta = get_fxdef_meta(int(fxid))
            base = f"fx{fxid}_{fxmeta['shortname']}_{fxmeta['chans']}".replace(",", "_")
            dim = fxval.get("dim")
            value = fxval.get("value")

            if value is None:
                continue

            if dim == "scalar":
                scalar_row[base] = value[0]
            elif dim == "1d":
                for i, v in enumerate(value):
                    scalar_row[f"{base}_{i}"] = v
            elif dim == "2d":
                for epoch_idx, vec in enumerate(value):
                    row = {"recording_id": recording_id, "epoch_id": epoch_idx}
                    for i, v in enumerate(vec):
                        row[f"{base}_f{i}"] = v
                    epoch_rows.append(row)
            
            print(f"[matrix]Successfully processed recording {recording_id} with {len(fx_values)} features")

        return scalar_row, epoch_rows
    except Exception as e:
        print(f"[matrix]Failed recording {recording_id}: {e}")
        return None, None
    finally:
        # Force garbage collection after each recording
        gc.collect()

def extract_feature_matrix(dataset_id: str, feature_set_id: str, output_dir: str = None, num_workers: int = None):
    if output_dir is None:
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(ROOT_DIR, "data", "processed")
        
    os.makedirs(output_dir, exist_ok=True)
    recording_ids = get_recording_ids_for_dataset(dataset_id)

    # For now, use single process to avoid multiprocessing issues
    print(f"Using single process mode to avoid multiprocessing issues")
    print(f"Processing {len(recording_ids)} recordings...")
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")

    scalar_rows = []
    epoch_rows = []
    processed_count = 0
    start_time = time.time()

    # Use single process approach to avoid multiprocessing issues
    for rid in recording_ids:
        try:
            print(f"[matrix]Processing recording {rid} ({processed_count + 1}/{len(recording_ids)})")
            result = process_one_recording((rid, feature_set_id, output_dir))
            processed_count += 1
            scalar_row, epochs = result
            
            if scalar_row:
                scalar_rows.append(scalar_row)
            if epochs:
                epoch_rows.extend(epochs)
                
            # Print progress with memory usage
            if processed_count % 5 == 0 or processed_count == len(recording_ids):
                elapsed = time.time() - start_time
                memory_usage = get_memory_usage()
                print(f"[matrix]Progress: {processed_count}/{len(recording_ids)} ({processed_count/len(recording_ids)*100:.1f}%) - {elapsed:.1f}s elapsed - {memory_usage:.1f} MB")
            
            # Force garbage collection more frequently
            if processed_count % 10 == 0:
                gc.collect()
                memory_after_gc = get_memory_usage()
                print(f"[matrix]Memory after GC: {memory_after_gc:.1f} MB")
                
        except Exception as e:
            print(f"[matrix]Failed recording {rid}: {e}")
            continue

    print(f"[matrix]All recordings processed successfully!")

    print(f"[matrix]Saving scalar matrix to {output_dir}")
    if scalar_rows:
        df_scalar = pd.DataFrame(scalar_rows)
        df_scalar.to_csv(os.path.join(output_dir, "feature_matrix_scalar.csv"), index=False)
        print(f"[matrix]Saved scalar matrix with {len(df_scalar)} rows and {len(df_scalar.columns)} columns")
    else:
        print("[matrix]Warning: No scalar data to save")

    if epoch_rows:
        df_2d = pd.DataFrame(epoch_rows)
        df_2d = df_2d.sort_values(["recording_id", "epoch_id"]).reset_index(drop=True)
        df_2d.to_csv(os.path.join(output_dir, "feature_matrix_2d.csv"), index=False)
        print(f"[matrix]Saved 2D matrix with {len(df_2d)} rows")
    else:
        print("[matrix]No 2D data to save")

    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    print(f"[matrix]Done! Processed {processed_count}/{len(recording_ids)} recordings in {total_time:.1f} seconds")
    print(f"[matrix]Final memory usage: {final_memory:.1f} MB")
    print(f"[matrix]Saved matrices to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--featureset", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    extract_feature_matrix(
        dataset_id=args.dataset,
        feature_set_id=args.featureset,
        num_workers=args.num_workers
    )
