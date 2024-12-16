import numpy as np
import os
from scipy.signal import find_peaks
import fnmatch
from pandas import read_excel
from scipy.optimize import curve_fit
import math
import re
from tqdm import tqdm
import random
import fnmatch

def az_el(df):
    # azimuth = np.arctan(df['CyclopeanEyeDirection.y'], df['CyclopeanEyeDirection.x'])
    azimuth = np.arctan(df['CyclopeanEyeDirection.x'], df['CyclopeanEyeDirection.z'])
    # elevation = np.arctan(df['CyclopeanEyeDirection.y'], np.sqrt(df['CyclopeanEyeDirection.x']**2 + df['CyclopeanEyeDirection.z']**2))
    elevation = np.arctan(df['CyclopeanEyeDirection.y'], np.sqrt(df['CyclopeanEyeDirection.x']**2 + df['CyclopeanEyeDirection.z']**2))
    df['CyclopeanEyeDirection.az'] = np.degrees(azimuth)
    df['CyclopeanEyeDirection.el'] = np.degrees(elevation)
    #mean offset?
    df['CyclopeanEyeDirection.az'] = df['CyclopeanEyeDirection.az'] - df['CyclopeanEyeDirection.az'].mean()
    df['CyclopeanEyeDirection.el'] = df['CyclopeanEyeDirection.el'] - df['CyclopeanEyeDirection.el'].mean()
    return df

def az_el_dot_world(df):
    azimuth = np.arctan(df['WorldDotPostion.x'], df['WorldDotPostion.z'])
    elevation = np.arctan(df['WorldDotPostion.y'], np.sqrt(df['WorldDotPostion.x']**2 + df['WorldDotPostion.z']**2))
    df['WorldDotPostion.az'] = np.degrees(azimuth)
    df['WorldDotPostion.el'] = np.degrees(elevation)
    #mean offset?
    df['WorldDotPostion.az'] = df['WorldDotPostion.az'] - df['WorldDotPostion.az'].mean()
    df['WorldDotPostion.el'] = df['WorldDotPostion.el'] - df['WorldDotPostion.el'].mean()
    return df

def az_el_dot_local(df):
    azimuth = np.arctan2(df['LocalDotPostion.x'], df['LocalDotPostion.z'])
    elevation = np.arctan2(df['LocalDotPostion.y'], np.sqrt(df['LocalDotPostion.x']**2 + df['LocalDotPostion.z']**2))
    df['LocalDotPostion.az'] = np.degrees(azimuth)
    df['LocalDotPostion.el'] = np.degrees(elevation)
    #mean offset?
    df['LocalDotPostion.az'] = df['LocalDotPostion.az'] - df['LocalDotPostion.az'].mean()
    df['LocalDotPostion.el'] = df['LocalDotPostion.el'] - df['LocalDotPostion.el'].mean()
    return df

def find_files(root_dir, substring):
    file_paths_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if substring in filename:
                folder_name = os.path.basename(dirpath)
                if folder_name not in file_paths_dict:
                    file_paths_dict[folder_name] = []
                file_paths_dict[folder_name].append(os.path.join(dirpath, filename))
    return file_paths_dict

def find_experiment_csv_files(root_dir):
    csv_file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                full_path = os.path.join(dirpath, filename)
                csv_file_paths.append(full_path)

    return csv_file_paths

def detect_square_wave_periods(df, column_name, min_length=30):

    low, high = np.percentile(df[column_name].dropna(),[25,52])
    
    # Detect large changes which are typical of square wave transitions
    peaks, _ = find_peaks(df[column_name].dropna().abs(),distance=10)#height=high,

    # Group peaks into periods of consistent large changes
    peaks = list(peaks)  # Convert array to list for easy manipulation
    i = 0
    while i < len(peaks) - 1:
        if peaks[i+1] - peaks[i] < min_length:
            peaks.pop(i)
        else:
            i += 1

    return peaks

def extract_con_and_control():
    main_directory = '2023_2024'

    src = []
    control = []

    for root, dirs, files in os.walk(main_directory):
        for dir_name in dirs:
            if 'CON1' in dir_name or 'C1' in dir_name:
                src.append(dir_name)
            elif not any(substring in dir_name for substring in ['CON', 'PC', 'C1', 'SF']):
                control.append(dir_name)

    # print("Concussions:")
    # print(src)

    # print("\nControls:")
    # print(control)

    return src, control

def find_csv_files_with_id(directory, substring):
    """
    search for csv files containing their ID, the file may or may not be in a subdirectory.
    """
    matching_files = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename, '*.csv') and substring in filename:
                matching_files.append(os.path.join(root, filename))

    return matching_files

def read_voms_data(file_path, sheet_name='VOMS'):
    """
    Reads data from a VOMS sheet in 2022-2023
    """
    try:
        # Load the specified sheet into a DataFrame
        df = read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def rmse(y_true, y_pred):
    """
    RMSE with handling of nans due to y_true having saccades removed
    """
    squared_diffs = (y_true - y_pred) ** 2
    
    #ignore nans in both arrays
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)

    mean_squared_diff = np.nanmean(squared_diffs[valid_indices])

    return np.sqrt(mean_squared_diff)

def is_sine_wave(series, threshold: float = 190):
    """
    Detect if a given pandas series is a sine wave.
    """
    #Normalize
    series = (series - series.mean()) / series.std()

    def sine_wave(x, amplitude, frequency, phase, offset):
        return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

    x_values = np.arange(len(series))

    frequencies = [1, 2, 3, 4] #hz
    best_mape = np.inf
    best_frequency = None

    for freq in frequencies:
        # Initial guess with varying frequency
        initial_guess = [1, freq / len(series), 0, 0]

        try:
            params, _ = curve_fit(sine_wave, x_values, series, p0=initial_guess)
            fitted_sine_wave = sine_wave(x_values, *params)

            mape = np.mean(np.abs((series - fitted_sine_wave) / series) * 100)

            # Check if this frequency gives a better fit
            if mape < best_mape:
                best_mape = mape
                best_frequency = freq

        except RuntimeError:
            continue

    # print(f'Best Frequency: {best_frequency} Hz, MAPE: {best_mape}')

    #True if best MAPE is below the threshold
    return best_mape < threshold, best_mape

def vor_asymmetry(df,direction):
    """
    VOR Asymmetry Index = ((VOR Gain_left + VOR Gain_right) / |VOR Gain_left - VOR Gain_right|) * 100
    """
    if direction == "azimuth":
        leftward = df[df['CyclopeanEyeDirection.az_filter'] > 0]
        rightward = df[df['CyclopeanEyeDirection.az_filter'] < 0]

        vor_gain_left = np.nanmean(leftward['CyclopeanEyeDirection.az_filter'] / leftward['HeadOrientation.y'])
        vor_gain_right = np.nanmean(rightward['CyclopeanEyeDirection.az_filter'] / rightward['HeadOrientation.y'])

        vor_gain_left = vor_gain_left[vor_gain_left <= 2]
        vor_gain_right = vor_gain_right[vor_gain_right <= 2]

        mean_vor_gain_left = np.nanmean(vor_gain_left)
        mean_vor_gain_right = np.nanmean(vor_gain_right)

        asymmetry_index = np.abs(mean_vor_gain_left - mean_vor_gain_right) / (mean_vor_gain_left + mean_vor_gain_right) * 100
    else:
        leftward = df[df['CyclopeanEyeDirection.el_filter'] > 0]
        rightward = df[df['CyclopeanEyeDirection.el_filter'] < 0]

        vor_gain_left = np.nanmean(leftward['CyclopeanEyeDirection.el_filter'] / leftward['HeadOrientation.x'])
        vor_gain_right = np.nanmean(rightward['CyclopeanEyeDirection.el_filter'] / rightward['HeadOrientation.x'])

        vor_gain_left = vor_gain_left[vor_gain_left <= 2]
        vor_gain_right = vor_gain_right[vor_gain_right <= 2]

        mean_vor_gain_left = np.nanmean(vor_gain_left)
        mean_vor_gain_right = np.nanmean(vor_gain_right)

        asymmetry_index = np.abs(mean_vor_gain_left - mean_vor_gain_right) / (mean_vor_gain_left + mean_vor_gain_right) * 100

    return asymmetry_index

def remove_dict_with_most_nans(data):
    """
    I got this from Perplexity
    """
    azimuth_dicts = [d for d in data if d.get('direction_az') == 'azimuth']
    elevation_dicts = [d for d in data if d.get('direction_el') == 'elevation']

    def count_nans(d):
        # Count the number of NaN values in the dictionary
        return sum(1 for v in d.values() if isinstance(v, float) and v != v)  # v != v checks for NaN

    # If there are multiple azimuth dicts, keep the one with the fewest NaNs
    if len(azimuth_dicts) > 1:
        azimuth_dicts.sort(key=count_nans)
        azimuth_dicts = azimuth_dicts[:1]

    # If there are multiple elevation dicts, keep the one with the fewest NaNs
    if len(elevation_dicts) > 1:
        elevation_dicts.sort(key=count_nans)
        elevation_dicts = elevation_dicts[:1]

    # Return the filtered list with one azimuth and one elevation dict
    return azimuth_dicts + elevation_dicts

def remove_substring(list_val):
    result = []
    for val in list_val:
        parts = val.split('_')
        if '_2' not in val and "_3" not in val:
            result.append(parts[0])
    return result


def find_experiment_csv_files_with_sf(root_dir,target_strings):
    csv_file_paths = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            #case-insensitive comparison
            if filename.lower().endswith('.csv'):
                # Check if the filename contains any target string followed by 'SF'
                if (target_strings in filename) and ("sf1" in filename.lower()) and "C1" not in filename:
                    if 'convergence' not in filename:
                        # print(target_strings)
                        # print(filename)
                        # input()
                        full_path = os.path.join(dirpath, filename)
                        csv_file_paths.append(full_path)
    return csv_file_paths

def find_experiment_csv_files_with_control(root_dir):
    csv_file_paths = []
    save_participants = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in tqdm(filenames):
            #case-insensitive comparison
            if filename.lower().endswith('.csv'):
                # Check if the filename contains any target string followed by 'SF'
                if ("sf1" not in filename.lower()) and ("C1" not in filename):
                    if 'convergence' not in filename:
                        extracted_id = re.search(r'pID_(.*?)_', filename).group(1)
                        if extracted_id != ' Test Emotibit2':
                            if extracted_id not in save_participants:
                                save_participants.append(extracted_id)
    #only wanty strings with 7 characters
    save_participants = [s for s in save_participants if len(s) == 7]
    #select 13 samples
    random_selection = random.sample(save_participants, 50)

    for root, dirs, files in os.walk(root_dir):
        for search_string in random_selection:
            for filename in fnmatch.filter(files, f'*{search_string}*'):
                if ("sf1" not in filename.lower()) and ("C1" not in filename):
                    if 'convergence' not in filename:
                        csv_file_paths.append(os.path.join(root, filename))
    return csv_file_paths, random_selection