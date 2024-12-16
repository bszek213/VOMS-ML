import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks, square, savgol_filter
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
from colorama import Fore, Style
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageSequence
from sys import argv
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from vr_voms_utils import find_files, az_el, detect_square_wave_periods, extract_con_and_control, find_experiment_csv_files, az_el_dot_local, az_el_dot_world, rmse
from vr_voms_utils import is_sine_wave, vor_asymmetry, remove_dict_with_most_nans, remove_substring, find_experiment_csv_files_with_sf, find_experiment_csv_files_with_control
import argparse
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import spearmanr, ttest_ind
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import re
from rapidfuzz import process
import traceback

plt.rcParams.update({
    'font.size': 12,                 # Slightly smaller for better readability in academic figures
    'font.weight': 'normal',         # Normal weight for body text
    'axes.labelsize': 14,            # Larger axis labels
    'axes.labelweight': 'bold',      # Bold axis labels for emphasis
    'axes.titlesize': 16,            # Larger title size
    'axes.titleweight': 'bold',      # Bold title for clarity
    'legend.fontsize': 12,           # Consistent legend size
    'legend.title_fontsize': 12,     # Consistent legend title size
    'xtick.labelsize': 12,           # Tick labels at a readable size
    'ytick.labelsize': 12,           # Tick labels at a readable size
    'lines.linewidth': 2,            # Thicker lines for clarity in figures
    'figure.titlesize': 16,          # Larger figure title
    'figure.titleweight': 'bold',    # Bold figure title for emphasis
    'savefig.dpi': 350,              # High resolution for publication-quality figures
    'figure.figsize': (8, 6)         # Standard size for figures
})

def fix_head_orientation(df, column='HeadOrientation.y'):
    angles = df[column].to_numpy()
    angles_rad = np.deg2rad(angles)
    # Unwrap the angles - helps remove the gimbal low of -180 - +180 sign flip
    unwrapped_angles_rad = np.unwrap(angles_rad)
    unwrapped_angles_deg = np.rad2deg(unwrapped_angles_rad)
    df[column] = unwrapped_angles_deg
    return df

def find_experiment_csv_files_src(root_dir='/media/brianszekely/TOSHIBA EXT/ExperimentData'):
    csv_file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if (filename.lower().endswith('.csv') and 
                ('c1' in filename.lower() or 'con' in filename.lower()) and
                not (any(f'sf{i}_' in filename.lower() for i in range(1, 6))) and 
                "convergence" not in filename.lower() and 
                "distance" not in filename.lower()):
                full_path = os.path.join(dirpath, filename)
                csv_file_paths.append(full_path)
    return csv_file_paths

def find_experiment_csv_files_healthy(root_dir='/media/brianszekely/TOSHIBA EXT/ExperimentData'):
    csv_file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if ((filename.lower().endswith('.csv')) and not (any(f'c{i}' in filename.lower() for i in range(1,6)))
                and not (any(f'SF{i}_' in filename.lower() for i in range(1,6))) and ("con" not in filename.lower()) and
                ("convergence" not in filename.lower()) and ("distance" not in filename.lower())):
                full_path = os.path.join(dirpath, filename)
                csv_file_paths.append(full_path)
    return csv_file_paths

class vrVomsSaccade():
    def __init__(self,input_files, subject_id='string') -> None:
        print('vrVoms Saccades Class')
        if input_files:
            self.experiment_files = input_files
            self.sub_id = subject_id
        else:
            for root, dirs, files in os.walk('/media/brianszekely/TOSHIBA EXT/ExperimentData'):
                if 'ExperimentData' in dirs:
                    self.prefix_external = os.path.join(root, 'ExperimentData')
            # self.prefix_external = '/media/brianszekely/TOSHIBA EXT/ExperimentData/'
            # self.all_files_dict = find_files(self.prefix_external, "experiment_data_pID")
            self.all_files = find_experiment_csv_files(self.prefix_external)
            self.experiment_files = []
            self.sub_id = subject_id
            subject_id = subject_id.split('_')[0]
            for string in self.all_files:
                if subject_id in string and 'experiment_data' in string:
                    self.experiment_files.append(string)
        
    def parse_condition_per(self,file):
            data = pd.read_csv(file)
            if data['Experiment'].str.contains("SACCADES", na=False).any(): #data['Experiment'].iloc[0] == "SACCADES":
                self.exp_df = data
                # plt.figure()
                # plt.plot(self.exp_df['CyclopeanEyeDirection.x'],label='hor')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.y'],label='vert')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.z'],label='z')
                # plt.legend()
                # plt.show()
                # plt.close()
                all_saccades = self.saccades(file)
                return all_saccades

    def saccades(self,file):
        self.exp_df = az_el(self.exp_df)
        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=11,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=11,polyorder=2)
        az_periods = detect_square_wave_periods(self.exp_df,'CyclopeanEyeDirection.az')
        saccades_dict, saccade_dir = self.saccade_heurstics()
        self.direction_sacc = saccade_dir
        print('=================')
        print(saccade_dir)
        print('=================')
        # plt.figure()
        # plt.plot(self.exp_df['CyclopeanEyeDirection.az'],label='hor')
        # plt.plot(self.exp_df['CyclopeanEyeDirection.el'],label='vert')
        # plt.legend()
        # plt.show()
        # plt.close()
        all_saccades = {}
        sacc_velocity_az, sacc_duration, sacc_velocity_el, sacc_amp_az, sacc_amp_el = [], [], [], [], []
        for keys, items in saccades_dict.items():
            filtered_df = self.exp_df[(self.exp_df['TimeStamp'] >= items[0]) & (self.exp_df['TimeStamp'] <= items[1])]

            saccade_vel_check_az = np.nanmean(filtered_df['CyclopeanEyeDirection.az_filter'].diff().abs()
                                                     / filtered_df["TimeStamp"].diff())
            saccade_vel_check_el = np.nanmean(filtered_df['CyclopeanEyeDirection.el_filter'].diff().abs()
                                                     / filtered_df["TimeStamp"].diff())
            sacc_amp_az.append((np.abs(filtered_df['CyclopeanEyeDirection.az_filter'].iloc[-1] - 
                            filtered_df['CyclopeanEyeDirection.az_filter'].iloc[0])) / 2) #divide by two to get the middle point as the start
            sacc_amp_el.append((np.abs(filtered_df['CyclopeanEyeDirection.el_filter'].iloc[-1] - 
                            filtered_df['CyclopeanEyeDirection.el_filter'].iloc[0])) / 2)
            # if saccade_vel_check_az >= 90 or saccade_vel_check_el >= 90:
            sacc_duration.append((items[1] - items[0]) * 1000)
            sacc_velocity_az.append(saccade_vel_check_az)
            sacc_velocity_el.append(saccade_vel_check_el)

        if saccade_dir == "horizontal":
            all_saccades['num_saccades_az'] = len(saccades_dict)
            all_saccades['num_saccades_el'] = np.nan
            all_saccades['sacc_duration_az'] = np.mean(sacc_duration)
            all_saccades['sacc_duration_el'] = np.nan
            all_saccades['sacc_velocity_az'] = np.mean([x for x in sacc_velocity_az if np.isfinite(x)])
            all_saccades['sacc_velocity_el'] = np.nan
            all_saccades['ID'] = self.exp_df['ParticipantID'].iloc[0]
            all_saccades['direction_az'] = saccade_dir
            all_saccades['direction_el'] = np.nan
            all_saccades['saccade_amp_az'] = np.mean(sacc_amp_az)
            all_saccades['saccade_amp_el'] = np.nan
        elif saccade_dir == "vertical":
            all_saccades['num_saccades_az'] = np.nan
            all_saccades['num_saccades_el'] = len(saccades_dict)
            all_saccades['sacc_duration_az'] = np.nan
            all_saccades['sacc_duration_el'] = np.mean(sacc_duration)
            all_saccades['sacc_velocity_az'] = np.nan
            all_saccades['sacc_velocity_el'] = np.mean([x for x in sacc_velocity_el if np.isfinite(x)])
            # all_saccades['experiment'] = self.exp_df['Experiment'].iloc[0]
            all_saccades['ID'] = self.exp_df['ParticipantID'].iloc[0]
            all_saccades['direction_az'] = np.nan
            all_saccades['direction_el'] = saccade_dir
            all_saccades['saccade_amp_az'] = np.nan
            all_saccades['saccade_amp_el'] = np.mean(sacc_amp_el)
        return all_saccades

    def saccade_heurstics(self):
        az_vel = np.abs(self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel = np.nan_to_num(az_vel, nan=0.0) #fill any nan with 0

        el_vel = np.abs(self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel = np.nan_to_num(el_vel, nan=0.0) #fill any nan with 0
        saccade_found = False
        saccades_dict = {}
        saccade_number = 1

        #check to see if this is horizonal or vertical saccade task
        # plt.figure()
        # plt.plot(self.exp_df['CyclopeanEyeDirection.az_filter'],label='az')
        # plt.plot(self.exp_df['CyclopeanEyeDirection.el_filter'],label='el')
        # plt.title(f"var az {np.var(self.exp_df['CyclopeanEyeDirection.az_filter'])} || var el {np.var(self.exp_df['CyclopeanEyeDirection.el_filter'])}")
        # plt.legend() 
        # plt.show()
        
        max_value_az = np.max(abs(self.exp_df['CyclopeanEyeDirection.az_filter']))
        max_value_el = np.max(abs(self.exp_df['CyclopeanEyeDirection.el_filter']))

        # Determine which column has the highest maximum value
        if max_value_az > max_value_el:
            max_value = max_value_az
            threshold = max_value * (85 / 100)
            
            near_max_count = np.sum(abs(self.exp_df['CyclopeanEyeDirection.az_filter']) >= threshold)
            percentage_az = (near_max_count / len(self.exp_df['CyclopeanEyeDirection.az_filter'])) * 100

            near_max_count = np.sum(abs(self.exp_df['CyclopeanEyeDirection.el_filter']) >= threshold)
            percentage_el = (near_max_count / len(self.exp_df['CyclopeanEyeDirection.el_filter'])) * 100
        else:
            max_value = max_value_el
            threshold = max_value * (85 / 100)
            
            near_max_count = np.sum(abs(self.exp_df['CyclopeanEyeDirection.az_filter']) >= threshold)
            percentage_az = (near_max_count / len(self.exp_df['CyclopeanEyeDirection.az_filter'])) * 100

            near_max_count = np.sum(abs(self.exp_df['CyclopeanEyeDirection.el_filter']) >= threshold)
            percentage_el = (near_max_count / len(self.exp_df['CyclopeanEyeDirection.el_filter'])) * 100

        # if self.sub_id == "LA22199C1_1":
        #     plt.figure()
        #     plt.plot(self.exp_df['CyclopeanEyeDirection.x'],label='az')
        #     plt.plot(self.exp_df['CyclopeanEyeDirection.y'],label='el')
        #     plt.title(f'percentage_az {percentage_az} | percentage_el {percentage_el}')
        #     plt.legend()
        #     plt.show()
        
        if percentage_az > percentage_el:
        # if np.max(self.exp_df['CyclopeanEyeDirection.az_filter']) > np.max(self.exp_df['CyclopeanEyeDirection.el_filter']):
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.az_filter']
            sacc_vel = az_vel
            saccade_dir = 'horizontal'
        else:
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.el_filter']
            sacc_vel = el_vel
            saccade_dir = 'vertical'
        for i in range(len(saccade_time_series)):
            if sacc_vel[i] > 30 and saccade_found == False: #velocity threshold for saccade
                saccade_start = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = True
            if sacc_vel[i] < 30 and saccade_found == True:
                saccade_end = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = False
                sub_df = self.exp_df[(self.exp_df['TimeStamp'] >= saccade_start) & 
                                     (self.exp_df['TimeStamp'] <= saccade_end)]
                # saccade_disp = abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[-1]) - abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[0])
                saccade_dur = (sub_df['TimeStamp'].iloc[-1] - sub_df['TimeStamp'].iloc[0]) * 1000
                # print(saccade_disp, saccade_dur*1000)
                if saccade_dur > 5:
                    saccades_dict[saccade_number] = [saccade_start,saccade_end]
                    saccade_number += 1
        return saccades_dict, saccade_dir
    
    def run_analysis(self):
        sacc_list = []
        for file_paths in tqdm(self.experiment_files):
            try:
                all_saccades = self.parse_condition_per(file_paths)
                if all_saccades:
                    sacc_list.append(all_saccades)
            except Exception as e:
                traceback.print_exc()

        return sacc_list

class vrVomsSP():
    def __init__(self,input_files,subject_id='None') -> None:
        print('vrVoms Smooth Pursuit Class')
        if input_files:
            self.experiment_files = input_files
            self.sub_id = subject_id
        else:
            for root, dirs, files in os.walk('/media'):
                if 'ExperimentData' in dirs:
                    self.prefix_external = os.path.join(root, 'ExperimentData')
            # self.prefix_external = '/media/brianszekely/TOSHIBA EXT/ExperimentData/'
            # self.all_files_dict = find_files(self.prefix_external, "experiment_data_pID")
            self.all_files = find_experiment_csv_files(self.prefix_external)
            self.experiment_files = []
            subject_id = subject_id.split('_')[0]
            for string in self.all_files:
                if subject_id in string and 'experiment_data' in string:
                    self.experiment_files.append(string)
        if len(self.experiment_files) < 1:
            print('=====================================')
            print(f'{subject_id} HAS NO DATA')
            print('=====================================')

    def parse_condition_per(self,file):
            data = pd.read_csv(file)
            if data['Experiment'].str.contains("SMOOTH_PURSUIT", na=False).any():#data['Experiment'].iloc[0] == "SMOOTH_PURSUIT":
                self.exp_df = data
                all_sp = self.smooth_pursuit(file)
                # plt.figure()
                # plt.plot(self.exp_df['CyclopeanEyeDirection.x'],label='hor')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.y'],label='vert')
                # plt.plot(self.exp_df['CyclopeanEyeDirection.z'],label='z')
                # plt.legend()
                # plt.show()
                # plt.close()
                return all_sp
            else:
                return None
            
    def smooth_pursuit(self,file):
        self.exp_df = az_el(self.exp_df)
        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=23,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=23,polyorder=2)
        self.exp_df['CyclopeanEyeDirection.el_filter'] = self.exp_df['CyclopeanEyeDirection.el_filter'] * 0.5
        sp_dict = self.sp_heuristics()
        return sp_dict

    
    def sp_heuristics(self):
        sp_dict = {}

        self.exp_df = az_el_dot_world(self.exp_df)
        self.exp_df = az_el_dot_local(self.exp_df)

        az_vel = np.abs(self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel = np.nan_to_num(az_vel, nan=0.0) #fill any nan with 0

        el_vel = np.abs(self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel = np.nan_to_num(el_vel, nan=0.0) #fill any nan with 0

        az_vel_dot = np.abs(self.exp_df['WorldDotPostion.az'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel_dot = np.nan_to_num(az_vel_dot, nan=0.0) #fill any nan with 0

        el_vel_dot = np.abs(self.exp_df['WorldDotPostion.el'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel_dot = np.nan_to_num(el_vel_dot, nan=0.0) #fill any nan with 0

        #L2 norm velocity
        norm_eye = np.sqrt((az_vel - el_vel)**2)
        norm_dot = np.sqrt((az_vel_dot - el_vel_dot)**2)

        #smooth out dot position
        sp_vel_dot = savgol_filter(norm_dot,window_length=23,polyorder=2)

        #detect saccades in sp
        saccades_dict = self.detect_saccades_in_sp()

        # fig, ax1 = plt.subplots(figsize=(15, 8),nrows=2,ncols=1)
        # ax1[0].plot(self.exp_df['CyclopeanEyeDirection.az_filter'].to_numpy(),
        #              self.exp_df['CyclopeanEyeDirection.el_filter'].to_numpy(),label='eye')
        # ax1[0].plot(self.exp_df['WorldDotPostion.az'].to_numpy(),
        #             self.exp_df['WorldDotPostion.el'].to_numpy(),label='dot')
        # ax1[1].plot(self.exp_df['TimeStamp'],norm_eye,label='norm eye')
        # ax1[1].plot(self.exp_df['TimeStamp'],sp_vel_dot,label='norm dot')

        # plt.title(mean_gain)
        # plt.legend()
        # plt.show()

        #cases where target_velocity is zero to avoid division by zero
        smooth_pursuit_gain = norm_eye / sp_vel_dot
        smooth_pursuit_gain = np.where(norm_eye == 0, 0, smooth_pursuit_gain)
        # if np.mean(smooth_pursuit_gain[smooth_pursuit_gain <= 2]) > 0.75:
        #     timestamps = self.exp_df['TimeStamp']
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(self.exp_df['CyclopeanEyeDirection.az_filter'], 
        #                 self.exp_df['CyclopeanEyeDirection.el_filter'], linewidth=3,
        #                 color='royalblue',
        #                 label='Velocity')
        #     for idx, (onset, offset) in saccades_dict.items():
        #         # Get the indices where the timestamps fall within the saccade onset and offset
        #         saccade_indices = np.where((timestamps >= onset) & (timestamps <= offset))[0]
        #         if len(saccade_indices) > 0:
        #             plt.scatter(self.exp_df['CyclopeanEyeDirection.az_filter'].iloc[saccade_indices], 
        #                         self.exp_df['CyclopeanEyeDirection.el_filter'].iloc[saccade_indices],c='orange')
        #     plt.xlabel('Azimuth (°)')
        #     plt.ylabel('Elevation (°)')
        #     plt.legend(loc='best')
        #     print(saccades_dict)
        #     plt.show()

        #change saccadic portions in the sp signal to NAN
        timestamps = self.exp_df['TimeStamp'].values
        for _, (start_time, end_time) in saccades_dict.items():
            #range of data
            indices_to_nan = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
            
            #set to nan
            norm_eye[indices_to_nan] == np.nan
            smooth_pursuit_gain[indices_to_nan] = np.nan

        #metrics
        smooth_pursuit_gain = smooth_pursuit_gain[smooth_pursuit_gain <= 2]

        mean_gain = np.nanmean(smooth_pursuit_gain)
        std_dev = np.nanstd(smooth_pursuit_gain)
        variability = std_dev / mean_gain
        rms_error = rmse(sp_vel_dot,norm_eye)

        #here is my range that I am making up: 0.5 and 1.1. based on a moderate range of 
        #possibilites, as normative gain is usually between 0.8 - 1.0
        print('======================')
        print(mean_gain)
        print('======================')
        # if mean_gain >= 0.25 and mean_gain <= 1.5:
        #Save data if condition met
        sp_dict['ID'] = self.exp_df['ParticipantID'].iloc[0]
        sp_dict['SP_gain'] = mean_gain
        sp_dict['SP_cv'] = variability
        sp_dict['SP_rmse'] = rms_error
        return sp_dict

    def detect_saccades_in_sp(self):
        az_vel = np.abs(self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel = np.nan_to_num(az_vel, nan=0.0) #fill any nan with 0

        el_vel = np.abs(self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        el_vel = np.nan_to_num(el_vel, nan=0.0) #fill any nan with 0

        saccade_found = False
        saccades_dict = {}
        saccade_number = 1

        #check to see if this is horizonal or vertical saccade task
        if np.var(self.exp_df['CyclopeanEyeDirection.az_filter']) > np.var(self.exp_df['CyclopeanEyeDirection.el_filter']):
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.az_filter']
            sacc_vel = az_vel
        else:
            saccade_time_series = self.exp_df['CyclopeanEyeDirection.el_filter']
            sacc_vel = el_vel

        for i in range(len(saccade_time_series)):
            if sacc_vel[i] > 30 and saccade_found == False: #velocity threshold for saccade
                saccade_start = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = True
            if sacc_vel[i] < 30 and saccade_found == True:
                saccade_end = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = False
                sub_df = self.exp_df[(self.exp_df['TimeStamp'] >= saccade_start) & 
                                     (self.exp_df['TimeStamp'] <= saccade_end)]
                # saccade_disp = abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[-1]) - abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[0])
                saccade_dur = (sub_df['TimeStamp'].iloc[-1] - sub_df['TimeStamp'].iloc[0]) * 1000
                # print(saccade_disp, saccade_dur*1000)
                if saccade_dur > 20:
                    saccades_dict[saccade_number] = [saccade_start,saccade_end]
                    saccade_number += 1

        return saccades_dict

    def run_analysis(self):
        sp_dict = []
        for file_paths in tqdm(self.experiment_files):
            try:
                all_sp = self.parse_condition_per(file_paths)
                if all_sp:
                    sp_dict.append(all_sp)
            except Exception as e:
                traceback.print_exc()
        return sp_dict

class vrVomsVOR():
    def __init__(self, input_files, subject_id='None') -> None:
        print('vrVoms VOR Class')
        if input_files:
            self.experiment_files = input_files
            self.sub_id = subject_id
            self.track_num_files = 0
        else:
            for root, dirs, files in os.walk('/media'):
                if 'ExperimentData' in dirs:
                    self.prefix_external = os.path.join(root, 'ExperimentData')
            # self.prefix_external = '/media/brianszekely/TOSHIBA EXT/ExperimentData/'
            # self.all_files_dict = find_files(self.prefix_external, "experiment_data_pID")
            self.all_files = find_experiment_csv_files(self.prefix_external)
            self.experiment_files = []
            self.sub_id = subject_id
            self.track_num_files = 0
            subject_id = subject_id.split('_')[0]
            for string in self.all_files:
                if subject_id in string and 'experiment_data' in string:
                    self.experiment_files.append(string)
        if len(self.experiment_files) < 1:
            print('=====================================')
            print(f'{subject_id} HAS NO DATA')
            print('=====================================')

    def parse_condition_per(self,file):
            data = pd.read_csv(file)
            if data['Experiment'].str.contains("VOR", na=False).any(): #data['Experiment'].iloc[0] == "VOR":
                self.exp_df = data
                all_vor = self.VOR(file)
                self.track_num_files += 1
                # if self.sub_id == "RT23267C1_1":
                #     print(all_vor)
                #     print(self.track_num_files)
                #     input()
                #     plt.figure()
                #     plt.plot(self.exp_df['CyclopeanEyeDirection.x'],label='hor')
                #     plt.plot(self.exp_df['CyclopeanEyeDirection.y'],label='vert')
                #     plt.plot(self.exp_df['HeadOrientation.y'],label='yaw')
                #     plt.plot(self.exp_df['HeadOrientation.x'],label='pitch')
                #     plt.legend()
                #     plt.show()
                #     plt.close()
                return all_vor
            else:
                return None
    
    def VOR(self,file):
        """
        elevation needs to be inversed
        find the average amount of lag between the head and eye data using find peaks
        phase shift the data backwards by that amount of samples. 
        fill the rest with interpolation
        """
        dict_vor = {}
        self.exp_df = az_el(self.exp_df)

        self.exp_df['HeadOrientation.x'] = self.exp_df['HeadOrientation.x'].apply(lambda x: x - 360 if x > 100 else x)
        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y'].apply(lambda x: x - 360 if x > 100 else x)
        self.exp_df['HeadOrientation.z'] = self.exp_df['HeadOrientation.z'].apply(lambda x: x - 360 if x > 100 else x)
        
        self.exp_df['HeadOrientation.x'] = self.exp_df['HeadOrientation.x'] - self.exp_df['HeadOrientation.x'].mean()
        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y'] - self.exp_df['HeadOrientation.y'].mean()
        self.exp_df['HeadOrientation.z'] = self.exp_df['HeadOrientation.z'] - self.exp_df['HeadOrientation.z'].mean()

        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(-self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=23,polyorder=2)
            self.exp_df['CyclopeanEyeDirection.el_filter'] = savgol_filter(self.exp_df['CyclopeanEyeDirection.el'],
                                                                window_length=23,polyorder=2)

        #correct for lag
        self.phase_align() 

        # if self.az_sine or self.el_sine:
        # if self.sub_id == "RT23267C1_1":
        #     plt.figure()
        #     plt.plot(self.exp_df['HeadOrientation.y'],label='yaw')
        #     plt.plot(self.exp_df['HeadOrientation.x'],label='pitch')
        #     plt.show()
        if np.max(self.exp_df['HeadOrientation.y']) > np.max(self.exp_df['HeadOrientation.x']):
            direction = 'azimuth'
        else:
            direction = 'elevation'

        #I hate doing this, but whomever coded the experiment did not handle missing data very well
        self.exp_df['CyclopeanEyeDirection.az_filter'] = self.exp_df['CyclopeanEyeDirection.az_filter'].interpolate(method='linear')
        self.exp_df['CyclopeanEyeDirection.el_filter'] = self.exp_df['CyclopeanEyeDirection.el_filter'].interpolate(method='linear')
        self.exp_df['HeadOrientation.x'] = self.exp_df['HeadOrientation.x'].interpolate(method='linear')
        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y'].interpolate(method='linear')

        self.exp_df['CyclopeanEyeDirection.az_filter'] = self.exp_df['CyclopeanEyeDirection.az_filter']
        self.exp_df['CyclopeanEyeDirection.el_filter'] = self.exp_df['CyclopeanEyeDirection.el_filter']
        self.exp_df['HeadOrientation.x'] = self.exp_df['HeadOrientation.x']
        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y']
        
        az_vel = self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        # az_vel = np.nan_to_num(az_vel, nan=0.0)
        el_vel = self.exp_df['CyclopeanEyeDirection.el_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        # el_vel = np.nan_to_num(el_vel, nan=0.0)

        pitch_vel = self.exp_df['HeadOrientation.x'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        # pitch_vel = np.nan_to_num(pitch_vel, nan=0.0)
        yaw_vel = self.exp_df['HeadOrientation.y'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        # yaw_vel = np.nan_to_num(yaw_vel, nan=0.0)

        finite_hor_ratio = np.abs(az_vel) / np.abs(yaw_vel)
        finite_hor_ratio = finite_hor_ratio[np.isfinite(finite_hor_ratio)]
        finite_hor_ratio = finite_hor_ratio[finite_hor_ratio <= 2]
        hor_gain = np.abs(np.nanmean(finite_hor_ratio))

        finite_ver_ratio = np.abs(el_vel) / np.abs(pitch_vel)
        finite_ver_ratio = finite_ver_ratio[np.isfinite(finite_ver_ratio)]
        finite_ver_ratio = finite_ver_ratio[finite_ver_ratio <= 2]
        ver_gain = np.abs(np.nanmean(finite_ver_ratio))
    
        #output variables
        #using possible range of VOR gain values 0.8 - 1.06
        print('====')
        print(hor_gain)
        print(ver_gain)
        print('====')
        if direction == "azimuth": #and (0.4 < hor_gain < 1.25)
            # dict_vor['vor_asym_az'] = vor_asymmetry(self.exp_df,direction)
            # dict_vor['vor_asym_el'] = np.nan
            dict_vor['az_vor'] = hor_gain
            dict_vor['el_vor'] = np.nan
            dict_vor['ID'] = self.exp_df['ParticipantID'].iloc[0]
            dict_vor['direction_az'] = direction
            dict_vor['direction_el'] = np.nan
            self.write_to_file(file, 'good_vor.txt')
        elif direction == "elevation": # and (0.4 < ver_gain < 1.25)
            # dict_vor['vor_asym_az'] = np.nan
            # dict_vor['vor_asym_el'] = vor_asymmetry(self.exp_df,direction)
            dict_vor['az_vor'] = np.nan
            dict_vor['el_vor'] = ver_gain
            dict_vor['ID'] = self.exp_df['ParticipantID'].iloc[0]
            dict_vor['direction_az'] = np.nan
            dict_vor['direction_el'] = direction
            # self.write_to_file(file, 'good_vor.txt')
        else:
            # self.write_to_file(file, 'bad_vor.txt')
            return dict_vor
        # else:
        #     self.write_to_file(file, 'bad_vor.txt')
        
        return dict_vor

    def phase_align(self):
        self.az_sine, self.mape_az = is_sine_wave(self.exp_df['HeadOrientation.y'],threshold=140)
        self.el_sine, self.mape_el = is_sine_wave(self.exp_df['HeadOrientation.x'],threshold=140)
        self.exp_df['CyclopeanEyeDirection.az_filter'] = self.exp_df['CyclopeanEyeDirection.az_filter'].shift(-7)
        self.exp_df['CyclopeanEyeDirection.el_filter'] = self.exp_df['CyclopeanEyeDirection.el_filter'].shift(-7)

    def write_to_file(self, file_name, target_file):
        try:
            with open(target_file, 'r') as f:
                lines = f.readlines()
                if file_name + '\n' in lines:
                    return
        except FileNotFoundError:
            pass

        with open(target_file, 'a') as f:
            f.write(file_name + '\n')

    def run_analysis(self):
        #concussion
        vor_dict_src = []
        for file_paths in tqdm(self.experiment_files):
            try:
                all_vor = self.parse_condition_per(file_paths)
                if all_vor:  
                    vor_dict_src.append(all_vor)
            except Exception as e:
                traceback.print_exc()
        # if self.track_num_files > 2:
        #     vor_dict_src = remove_dict_with_most_nans(vor_dict_src)
            # print(vor_dict_src)
            # input()
        return vor_dict_src

class vrVomsVMS():
    def __init__(self, input_files, subject_id='None') -> None:
        print('vrVoms VMS Class')
        if input_files:
            self.experiment_files = input_files
            self.sub_id = subject_id
        else:
            for root, dirs, files in os.walk('/media'):
                if 'ExperimentData' in dirs:
                    self.prefix_external = os.path.join(root, 'ExperimentData')
            # self.prefix_external = '/media/brianszekely/TOSHIBA EXT/ExperimentData/'
            # self.all_files_dict = find_files(self.prefix_external, "experiment_data_pID")
            self.all_files = find_experiment_csv_files(self.prefix_external)
            self.experiment_files = []
            subject_id = subject_id.split('_')[0]
            for string in self.all_files:
                if subject_id in string and 'experiment_data' in string:
                    self.experiment_files.append(string)
        if len(self.experiment_files) < 1:
            print('=====================================')
            print(f'{subject_id} HAS NO DATA')
            print('=====================================')

    def parse_condition_per(self,file):
            data = pd.read_csv(file)
            if data['Experiment'].str.contains("VMS", na=False).any(): #data['Experiment'].iloc[0] == "VMS":
                self.exp_df = data
                all_vor = self.VMS(file)
                return all_vor
            else:
                return None
            
    def VMS(self,file):
        vms_dict = {}
        self.exp_df = az_el(self.exp_df)
        self.exp_df = az_el_dot_world(self.exp_df)
        self.exp_df = az_el_dot_local(self.exp_df)

        #remove gimbal lock
        self.exp_df = fix_head_orientation(self.exp_df,column='HeadOrientation.y')

        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y'] - self.exp_df['HeadOrientation.y'].mean()

        if 'CyclopeanEyeDirection.az_filter' not in self.exp_df.columns:
            self.exp_df['CyclopeanEyeDirection.az_filter'] = savgol_filter(-self.exp_df['CyclopeanEyeDirection.az'],
                                                                    window_length=23,polyorder=2)
        #phase align due to lag
        self.exp_df['CyclopeanEyeDirection.az_filter'] = self.exp_df['CyclopeanEyeDirection.az_filter'].interpolate(method='linear')
        self.exp_df['HeadOrientation.y'] = self.exp_df['HeadOrientation.y'].interpolate(method='linear')
        self.exp_df['CyclopeanEyeDirection.az_filter'] = self.exp_df['CyclopeanEyeDirection.az_filter'].shift(-7)

        #output metrics
        vms_dict['vms_asymmetry'] = self.vms_asymmetry()
        vms_dict['num_saccades_vms'] = self.detect_saccades_in_vms()
        vms_dict['vms_mape'] = self.gaze_position_error()
        vms_dict['vel_ratio'] = self.peak_vel_ratio()
        vms_dict['ID'] = self.exp_df['ParticipantID'].iloc[0]

        return vms_dict

    def peak_vel_ratio(self):
        eye_velocity = self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        head_velocity = self.exp_df['HeadOrientation.y'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))

        eye_velocity = eye_velocity[~np.isnan(eye_velocity)]
        head_velocity = head_velocity[~np.isnan(head_velocity)]

        peak_eye_velocity = np.max(np.abs(eye_velocity))
        peak_head_velocity = np.max(np.abs(head_velocity))

        return peak_eye_velocity / peak_head_velocity
    
    def gaze_position_error(self):
        """
        Big issue here as the controllers are technically the target, which is not fixed in the head.
        Technically speaking, the eye data should have 0 eye velocity, which means perfect fixation
        """
        eye_velocity = self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        eye_velocity = eye_velocity[~np.isnan(eye_velocity)]
        absolute_percentage_error = np.abs(eye_velocity)
        mape = np.mean(absolute_percentage_error) * 100
        return mape

    def detect_saccades_in_vms(self):
        saccade_found = False
        saccades_dict = {}
        saccade_number = 0

        #check to see if this is horizonal or vertical saccade task
        saccade_time_series = self.exp_df['CyclopeanEyeDirection.az_filter']
        az_vel = np.abs(self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp'])))
        az_vel = np.nan_to_num(az_vel, nan=0.0) #fill any nan with 0
        sacc_vel = az_vel

        #anything above SP should be "Saccadic"
        for i in range(len(saccade_time_series)):
            if sacc_vel[i] > 50 and saccade_found == False: #velocity threshold for saccade
                saccade_start = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = True
            if sacc_vel[i] < 50 and saccade_found == True:
                saccade_end = self.exp_df['TimeStamp'].iloc[i]
                saccade_found = False
                sub_df = self.exp_df[(self.exp_df['TimeStamp'] >= saccade_start) & 
                                    (self.exp_df['TimeStamp'] <= saccade_end)]
                # saccade_disp = abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[-1]) - abs(sub_df['CyclopeanEyeDirection.az_filter'].iloc[0])
                saccade_dur = (sub_df['TimeStamp'].iloc[-1] - sub_df['TimeStamp'].iloc[0]) * 1000
                # print(saccade_disp, saccade_dur*1000)
                if saccade_dur > 20:
                    saccades_dict[saccade_number] = [saccade_start,saccade_end]
                    saccade_number += 1
        return saccade_number

    def vms_asymmetry(self):
        eye_velocity = self.exp_df['CyclopeanEyeDirection.az_filter'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        head_velocity = self.exp_df['HeadOrientation.y'].diff().to_numpy() / np.mean(np.diff(self.exp_df['TimeStamp']))
        negative = head_velocity < 0
        positive = head_velocity > 0
        negative_eye_vel = np.nanmean(np.abs(eye_velocity[negative]))
        positive_eye_vel = np.nanmean(np.abs(eye_velocity[positive]))
        asymmetry = (positive_eye_vel - negative_eye_vel) / (positive_eye_vel + negative_eye_vel)
        return asymmetry * 100  #percentage
    
    def run_analysis(self):
        #concussion
        vms_list_src = []
        for file_paths in self.experiment_files:
            try:
                all_vms = self.parse_condition_per(file_paths)
                if all_vms:
                    vms_list_src.append(all_vms)
            except Exception as e:
                traceback.print_exc()
        return vms_list_src


def main():
    src_files = find_experiment_csv_files_src()
    healthy_files = find_experiment_csv_files_healthy()

    print('======= Saccades =======')
    if not os.path.exists('Saccades_SRC.csv'):
        #Saccades - SRC
        sacc_list = vrVomsSaccade(input_files=src_files).run_analysis()
        df_sacc_src = pd.DataFrame(sacc_list)
        #fill NaNs in adjacent rows with the same ID
        for index in range(len(df_sacc_src) - 1):
            if df_sacc_src.loc[index, 'ID'] == df_sacc_src.loc[index + 1, 'ID']:
                for col in df_sacc_src.columns:
                    if pd.isna(df_sacc_src.loc[index, col]) and not pd.isna(df_sacc_src.loc[index + 1, col]):
                        df_sacc_src.loc[index, col] = df_sacc_src.loc[index + 1, col]
                    elif pd.isna(df_sacc_src.loc[index + 1, col]) and not pd.isna(df_sacc_src.loc[index, col]):
                        df_sacc_src.loc[index + 1, col] = df_sacc_src.loc[index, col]

        #remove redundant rows
        df_sacc_src = df_sacc_src.loc[~df_sacc_src.duplicated(subset='ID', keep='first')].reset_index(drop=True)

        #fill na with mean to preserve as much data as possible
        for col in df_sacc_src.columns:
            if pd.api.types.is_numeric_dtype(df_sacc_src[col]):
                df_sacc_src[col].fillna(df_sacc_src[col].mean(), inplace=True)
        # df_sacc_src[df_sacc_src.select_dtypes(include='number').columns] = df_sacc_src.select_dtypes(include='number').fillna(df_sacc_src.select_dtypes(include='number').mean())

        #Saccades - CONTROL
        sacc_list = vrVomsSaccade(input_files=healthy_files).run_analysis()
        df_sacc_healthy = pd.DataFrame(sacc_list)
        #fill NaNs in adjacent rows with the same ID
        for index in range(len(df_sacc_healthy) - 1):
            if df_sacc_healthy.loc[index, 'ID'] == df_sacc_healthy.loc[index + 1, 'ID']:
                for col in df_sacc_healthy.columns:
                    if pd.isna(df_sacc_healthy.loc[index, col]) and not pd.isna(df_sacc_healthy.loc[index + 1, col]):
                        df_sacc_healthy.loc[index, col] = df_sacc_healthy.loc[index + 1, col]
                    elif pd.isna(df_sacc_healthy.loc[index + 1, col]) and not pd.isna(df_sacc_healthy.loc[index, col]):
                        df_sacc_healthy.loc[index + 1, col] = df_sacc_healthy.loc[index, col]

        #remove redundant rows
        df_sacc_healthy = df_sacc_healthy.loc[~df_sacc_healthy.duplicated(subset='ID', keep='first')].reset_index(drop=True)
        #fill na with mean to preserve as much data as possible
        for col in df_sacc_healthy.columns:
            if pd.api.types.is_numeric_dtype(df_sacc_healthy[col]):
                df_sacc_healthy[col].fillna(df_sacc_healthy[col].mean(), inplace=True)
        # df_sacc_healthy[df_sacc_healthy.select_dtypes(include='number').columns] = df_sacc_healthy.select_dtypes(include='number').fillna(df_sacc_healthy.select_dtypes(include='number').mean())
        print(df_sacc_healthy)

        df_sacc_src.to_csv('Saccades_SRC.csv',index=False)
        df_sacc_healthy.to_csv('Saccades_Control.csv',index=False)

    print('======= Smooth Pursuit =======')
    if not os.path.exists('SP_SRC.csv'):
        sp_list_src = vrVomsSP(input_files=src_files).run_analysis()
        df_sp_src = pd.DataFrame(sp_list_src)

        sp_list_healthy = vrVomsSP(input_files=healthy_files).run_analysis()
        df_sp_healthy = pd.DataFrame(sp_list_healthy)

        for col in df_sp_src.columns:
            if pd.api.types.is_numeric_dtype(df_sp_src[col]):
                df_sp_src[col].fillna(df_sp_src[col].mean(), inplace=True)
        for col in df_sp_healthy.columns:
            if pd.api.types.is_numeric_dtype(df_sp_healthy[col]):
                df_sp_healthy[col].fillna(df_sp_healthy[col].mean(), inplace=True)
        
        df_sp_src.to_csv('SP_SRC.csv',index=False)
        df_sp_healthy.to_csv('SP_Control.csv',index=False)
    print('======= VOR =======')
    if not os.path.exists('VOR_SRC.csv'):
        vor_list_src = vrVomsVOR(input_files=src_files).run_analysis()
        df_vor_src = pd.DataFrame(vor_list_src)

        for index in range(len(df_vor_src) - 1):
            if df_vor_src.loc[index, 'ID'] == df_vor_src.loc[index + 1, 'ID']:
                for col in df_vor_src.columns:
                    if pd.isna(df_vor_src.loc[index, col]) and not pd.isna(df_vor_src.loc[index + 1, col]):
                        df_vor_src.loc[index, col] = df_vor_src.loc[index + 1, col]
                    elif pd.isna(df_vor_src.loc[index + 1, col]) and not pd.isna(df_vor_src.loc[index, col]):
                        df_vor_src.loc[index + 1, col] = df_vor_src.loc[index, col]

        #remove redundant rows
        df_vor_src = df_vor_src.loc[~df_vor_src.duplicated(subset='ID', keep='first')].reset_index(drop=True)

        vor_list_healthy = vrVomsVOR(input_files=healthy_files).run_analysis()
        df_vor_healthy = pd.DataFrame(vor_list_healthy)

        for index in range(len(df_vor_healthy) - 1):
            if df_vor_healthy.loc[index, 'ID'] == df_vor_healthy.loc[index + 1, 'ID']:
                for col in df_vor_healthy.columns:
                    if pd.isna(df_vor_healthy.loc[index, col]) and not pd.isna(df_vor_healthy.loc[index + 1, col]):
                        df_vor_healthy.loc[index, col] = df_vor_healthy.loc[index + 1, col]
                    elif pd.isna(df_vor_healthy.loc[index + 1, col]) and not pd.isna(df_vor_healthy.loc[index, col]):
                        df_vor_healthy.loc[index + 1, col] = df_vor_healthy.loc[index, col]

        #remove redundant rows
        df_vor_healthy = df_vor_healthy.loc[~df_vor_healthy.duplicated(subset='ID', keep='first')].reset_index(drop=True)

        for col in df_vor_src.columns:
            if pd.api.types.is_numeric_dtype(df_vor_src[col]):
                df_vor_src[col].fillna(df_vor_src[col].mean(), inplace=True)
        for col in df_vor_healthy.columns:
            if pd.api.types.is_numeric_dtype(df_vor_healthy[col]):
                df_vor_healthy[col].fillna(df_vor_healthy[col].mean(), inplace=True)

        df_vor_src.to_csv('VOR_SRC.csv',index=False)
        df_vor_healthy.to_csv('VOR_Control.csv',index=False)

        print('======= VMS =======')
    if not os.path.exists('VMS_SRC.csv'):
        df_vms_src = pd.DataFrame(vrVomsVMS(input_files=src_files).run_analysis())
        df_vms_healthy = pd.DataFrame(vrVomsVMS(input_files=healthy_files).run_analysis())

        for col in df_vms_src.columns:
            if pd.api.types.is_numeric_dtype(df_vms_src[col]):
                df_vms_src[col].fillna(df_vms_src[col].mean(), inplace=True)
        for col in df_vms_healthy.columns:
            if pd.api.types.is_numeric_dtype(df_vms_healthy[col]):
                df_vms_healthy[col].fillna(df_vms_healthy[col].mean(), inplace=True)

        df_vms_src.to_csv('VMS_SRC.csv',index=False)
        df_vms_healthy.to_csv('VMS_Control.csv',index=False)

if __name__ == "__main__":
    main()