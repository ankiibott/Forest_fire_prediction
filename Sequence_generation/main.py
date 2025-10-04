import os
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from concurrent.futures import ThreadPoolExecutor

# --- 1. Global Configuration ---
# Variables are defined here, matching your notebook cells 16 and 17 definitions.
SEQ_LEN = 6                 
HORIZONS = 3               
PATCH_SIZE = 13             
PATCH_H = PATCH_W = PATCH_SIZE
CHANNELS = 7
HALF = PATCH_SIZE // 2
FILL_NAN_VALUE = 0.0

REQUIRED_COLS = [
    "era5_t2m_file", "era5_d2m_file", "era5_tp_file",
    "era5_u10_file", "era5_v10_file",
    "viirs_file", "dem_file", "lulc_file"
]
BATCH_SIZE = 16

# --- 2. Data Loading and Patch Extraction Functions ---

def _load_single_raster(path):
    """Loads a single raster file from the given path."""
    with rasterio.open(path) as src:
        arr = src.read() 
    # Return 2D array if single band, otherwise 3D
    return arr[0] if arr.shape[0] == 1 else arr

def load_rasters(df, raster_cols, max_workers=8):
    """Loads all unique raster paths in parallel and caches them."""
    all_paths = set()
    for col in raster_cols:
        if col in df.columns:
            all_paths.update(df[col].dropna().unique())
    all_paths = list(all_paths)
    
    cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_load_single_raster, all_paths))

    for path, arr in zip(all_paths, results):
        if arr is not None:
            cache[path] = arr
    return cache

def _safe_center(h, w, patch_size=PATCH_SIZE):
    """Calculates a safe center (r, c) for patch extraction."""
    half = patch_size // 2
    r = np.clip(h // 2, half, h - half - 1)
    c = np.clip(w // 2, half, w - half - 1)
    return r, c

def _extract_patch(arr, row, col, patch_size=PATCH_SIZE):
    """Extracts a patch from a 2D array, zero-padding near edges."""
    half = patch_size // 2
    h, w = arr.shape
    
    r0, r1 = row - half, row + half + 1
    c0, c1 = col - half, col + half + 1
    
    patch = np.zeros((patch_size, patch_size), dtype=arr.dtype)
    
    # Clip coordinates to array bounds
    r0_clip, r1_clip = max(r0, 0), min(r1, h)
    c0_clip, c1_clip = max(c0, 0), min(c1, w)
    
    # Coordinates in the patch (pr, pc)
    pr0, pr1 = r0_clip - r0, (r0_clip - r0) + (r1_clip - r0_clip)
    pc0, pc1 = c0_clip - c0, (c0_clip - c0) + (c1_clip - c0_clip)
    
    # Copy data from array to patch
    patch[pr0:pr1, pc0:pc1] = arr[r0_clip:r1_clip, c0_clip:c1_clip]
    return patch

def build_sample(seq_rows, horizon_rows, cache, force_fire=False):
    """Constructs a single (X, y) sample from sequence and horizon rows."""
    
    # Build X (Input Sequence)
    seq_patches = []
    for _, row in seq_rows.iterrows():
        bands = []
        # Era5 variables (5 bands)
        for var in ["era5_t2m_file", "era5_d2m_file", "era5_tp_file", "era5_u10_file", "era5_v10_file"]:
            arr = cache[row[var]]
            if len(arr.shape) == 3: arr = arr[0]
            h, w = arr.shape
            r, c = _safe_center(h, w)
            bands.append(_extract_patch(arr, r, c))
        
        # DEM and LULC (2 bands)
        dem = cache[row["dem_file"]]
        lulc = cache[row["lulc_file"]]
        if len(dem.shape) == 3: dem = dem[0]
        if len(lulc.shape) == 3: lulc = lulc[0]
        h, w = dem.shape
        r, c = _safe_center(h, w)
        bands.append(_extract_patch(dem, r, c))
        bands.append(_extract_patch(lulc, r, c))
        
        seq_patches.append(np.stack(bands, axis=-1))
    X = np.stack(seq_patches, axis=0)
    
    # Build y (Target Sequence)
    horizon_patches = []
    for _, row in horizon_rows.iterrows():
        viirs_stack = cache[row["viirs_file"]]
        
        # Extract target band index from the row (as done in your notebook)
        target_band_idx_list = eval(row["target_band_idxs"])
        idx = target_band_idx_list[0]
        band = viirs_stack[idx - 1] # assuming viirs_stack is 3D and contains multiple time steps
        
        h, w = band.shape
        r, c = _safe_center(h, w)

        # Force fire logic (only if needed for specialized sampling, typically ignored for main flow)
        if force_fire and np.any(band > 0):
            fire_pos = np.argwhere(band > 0)
            r, c = fire_pos[np.random.randint(len(fire_pos))]

        horizon_patches.append(_extract_patch(band, r, c))
    y = np.stack(horizon_patches, axis=0)

    return X.astype("float32"), y.astype("float32")


# --- 3. Generator and Dataset Functions ---

def make_generator(df, cache, fire_ratio=0.5):
    """Generator function for balanced sampling of fire and non-fire events."""
    valid_start_indices = list(range(len(df) - SEQ_LEN - HORIZONS + 1))
    fire_start_indices = []
    non_fire_start_indices = []
    
    print("Scanning data for fire and non-fire events...")
    for i in valid_start_indices:
        horizon_rows = df.iloc[i + SEQ_LEN : i + SEQ_LEN + HORIZONS]
        # This checks for fire in *any* horizon time step
        has_fire = any(np.any(cache[row["viirs_file"]] > 0) for _, row in horizon_rows.iterrows())
        
        if has_fire:
            fire_start_indices.append(i)
        else:
            non_fire_start_indices.append(i)

    num_fire_samples = len(fire_start_indices)
    
    if num_fire_samples == 0:
        print("Warning: No fire events found in the dataset.")
        num_non_fire_samples_to_use = min(len(non_fire_start_indices), 1000) 
    else:
        # Calculate samples needed for the desired fire_ratio
        num_non_fire_samples_to_use = int((num_fire_samples / fire_ratio) - num_fire_samples)
        num_non_fire_samples_to_use = min(num_non_fire_samples_to_use, len(non_fire_start_indices))

    fire_indices_to_use = fire_start_indices

    if len(non_fire_start_indices) > 0 and num_non_fire_samples_to_use > 0:
      non_fire_indices_to_use = np.random.choice(
          non_fire_start_indices,
          size=num_non_fire_samples_to_use,
          replace=False 
      )
      indices_to_use = np.concatenate([fire_indices_to_use, non_fire_indices_to_use])
    else:
      indices_to_use = np.array(fire_indices_to_use)

    np.random.shuffle(indices_to_use)
    indices_to_use = indices_to_use.astype(int)
    
    print(f"Generator initialized. Found {len(fire_indices_to_use)} fire samples and using {len(indices_to_use) - len(fire_indices_to_use)} non-fire samples.")

    for i in indices_to_use:
        seq_rows = df.iloc[i : i + SEQ_LEN]
        horizon_rows = df.iloc[i + SEQ_LEN : i + SEQ_LEN + HORIZONS]
        X, y = build_sample(seq_rows, horizon_rows, cache)
        yield X, y

def create_dataset(df, cache, shuffle=True, fire_ratio=0.5, shuffle_buf=256):
    """Creates a tf.data.Dataset from the generator."""
    output_signature = (
        tf.TensorSpec(shape=(SEQ_LEN, PATCH_SIZE, PATCH_SIZE, CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(HORIZONS, PATCH_SIZE, PATCH_SIZE), dtype=tf.float32),
    )
    
    ds = tf.data.Dataset.from_generator(
        lambda: make_generator(df, cache, fire_ratio=fire_ratio),
        output_signature=output_signature
    )
    
    if shuffle:
        ds = ds.shuffle(shuffle_buf, reshuffle_each_iteration=True)
    
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --- 4. Model Architecture (With Lambda Fix) ---

# --- Fix: Define Lambda Functions with explicit output_shape ---

def slice_output_func(x):
    return x[:, :HORIZONS, :, :, :]

def slice_output_shape(input_shape):
    # Input shape: (Batch, 6, 13, 13, 1) -> Output shape: (Batch, 3, 13, 13, 1)
    return (input_shape[0], HORIZONS, input_shape[2], input_shape[3], input_shape[4])

def squeeze_output_func(x):
    return tf.squeeze(x, axis=-1)

def squeeze_output_shape(input_shape):
    # Input shape: (Batch, 3, 13, 13, 1) -> Output shape: (Batch, 3, 13, 13)
    return input_shape[:-1] 

def build_conv_lstm_unet_model(
    seq_len=SEQ_LEN,
    patch_h=PATCH_H,
    patch_w=PATCH_W,
    channels=CHANNELS,
    horizons=HORIZONS
):
    inp = layers.Input(shape=(seq_len, patch_h, patch_w, channels))

    # ENCODER 1
    enc1 = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(inp)
    enc1_pool = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(enc1)

    # ENCODER 2
    enc2 = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(enc1_pool)
    enc2_pool = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(enc2)

    # BOTTLENECK
    bottleneck = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(enc2_pool)

    # DECODER 1
    dec1_up = layers.UpSampling3D(size=(1, 2, 2))(bottleneck)
    dec1_up = layers.Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu')(dec1_up)
    dec1_up_cropped = layers.Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))(dec1_up)
    dec1_concat = layers.Concatenate(axis=-1)([dec1_up_cropped, enc2])

    # DECODER 2
    dec2_up = layers.UpSampling3D(size=(1, 2, 2))(dec1_concat)
    dec2_up = layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation='relu')(dec2_up)
    dec2_up_cropped = layers.Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))(dec2_up)
    dec2_concat = layers.Concatenate(axis=-1)([dec2_up_cropped, enc1])

    # OUTPUT CONVLSTM
    output_convlstm = layers.ConvLSTM2D(
        filters=1, kernel_size=(3, 3), padding='same', return_sequences=True, activation='sigmoid'
    )(dec2_concat)

    # FIX 1: Slicing Lambda Layer
    output_sliced = layers.Lambda(
        slice_output_func, 
        output_shape=slice_output_shape,
        name='output_slicer'
    )(output_convlstm)

    # FIX 2: Squeeze Lambda Layer
    final_output = layers.Lambda(
        squeeze_output_func,
        output_shape=squeeze_output_shape,
        name='final_squeeze'
    )(output_sliced)
    
    model = models.Model(inputs=inp, outputs=final_output)
    return model

# --- 5. Main Execution and Training ---

if __name__ == "__main__":
    # --- Data Setup ---
    # NOTE: You must ensure this CSV path is correct in your environment
    csv_path = r"C:\Users\Ankit\Datasets_Forest_fire\sequence_index_hourly_binary.csv"
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {csv_path}. Please check the path.")
        exit()

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    TOTAL = len(df)
    VAL_SPLIT = 0.2
    val_size = int(TOTAL * VAL_SPLIT)

    val_df = df.iloc[:val_size].copy()
    train_df = df.iloc[val_size:].copy()

    print(f"Total samples: {TOTAL}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    raster_cols = REQUIRED_COLS
    print("Loading rasters into memory...")
    cache = load_rasters(df, raster_cols, max_workers=8)
    print(f"Loaded {len(cache)} rasters into memory ✅")

    # Create Datasets
    # fire_ratio=0.5 means 50% fire events, 50% non-fire events
    train_dataset = create_dataset(train_df, cache, fire_ratio=0.5)
    val_dataset = create_dataset(val_df, cache, fire_ratio=0.5)
    
    # --- Model Compilation ---
    model = build_conv_lstm_unet_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    )
    
    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE
    
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # --- Callbacks ---
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    # Recommended to save in the modern .keras format
    checkpoint = callbacks.ModelCheckpoint(
        "best_unet_model.keras", 
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # --- Training ---
    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[early_stop, checkpoint],
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )