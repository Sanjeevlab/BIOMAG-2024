clc
clear 
close all
%% %%%%%%%%%%%%%% Step 1 %%%%%%%%%%%%%%%%%%%%%%
% PATH SET-UP
restoredefaultpath % restore default folder for matlab

maindir = pwd;     % keep main path
%set up the path of fieldtrip, this differs from computer to computer
cd('/home/sanjeev/MATLAB Add-Ons/Collections/fieldtrip-2021')
%%cd('C:\Users\Admin\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\FieldTrip')

addpath(pwd)
cd(maindir)        % return to main
ft_defaults
%% %%%%%%%%%%%%%% Step 2 %%%%%%%%%%%%%%%%%%%%%%
% Artefact Removal Filtering using max filtering done in mne python....

%% %%%%%%%%%%%%%% Step 3 %%%%%%%%%%%%%%%%%%%%%%
data_path = '/home/sanjeev/Desktop/BIO/MEG/sub-CA103/ses-1/meg/raw_sss/';
file_names = {'sub-01_ses-01_task-dur_run-01_raw_sss.fif','sub-01_ses-01_task-dur_run-02_raw_sss.fif','sub-01_ses-01_task-dur_run-03_raw_sss.fif'} 
%% 
dataset = '/home/sanjeev/Desktop/BIO/MEG/sub-CA103/ses-1/meg/raw_sss/sub-01_ses-01_task-dur_run-01_raw_sss.fif'
%% 
event = ft_read_event(dataset)
%'sub-01_ses-01_task-dur_run-02_raw_sss.fif', ...
%    'sub-01_ses-01_task-dur_run-03_raw_sss.fif', ...
%    'sub-01_ses-01_task-dur_run-04_raw_sss.fif', ...
%    'sub-01_ses-01_task-dur_run-05_raw_sss.fif'}
%% 
event.type
%% step 13 READING EVENTS

%% step 14 trail function
cfg                    = [];
cfg.trialfun = 'ft_trialfun_general';
cfg.trialdef.prestim   = 2;                   % in seconds
cfg.trialdef.poststim  = 2;                   % in seconds
cfg.trialdef.eventtype = ['STI101'];            % get a list of the available types
cfg.trialdef.eventvalue = [201 202 203];
cfg.dataset            = dataset;             % set the name of the dataset
cfg_tr_def             = ft_definetrial(cfg);

cfg.channel            = 'MEG';             % define channel type
data1                   = ft_preprocessing(cfg_tr_def); % read raw data
%% 
data1
