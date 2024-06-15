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
data_path = '/home/sanjeev/Desktop/BIO/MEG/sub-CA103/ses-1/meg/raw_sss/';
%% 
dataset{1,1}='sub-01_ses-01_task-dur_run-01_raw_sss.fif';
dataset{1,2}='sub-01_ses-01_task-dur_run-02_raw_sss.fif';
dataset{1,3}='sub-01_ses-01_task-dur_run-03_raw_sss.fif';
dataset{1,4}='sub-01_ses-01_task-dur_run-04_raw_sss.fif';
dataset{1,5}='sub-01_ses-01_task-dur_run-05_raw_sss.fif';
%%
for i=1:length(dataset)
    cfg  = [];
    cfg.dataset = [data_path,char(dataset{1,i})];  
    cfg.trialfun = 'ft_trialfun_general';  
    cfg.trialdef.eventtype  = 'STI101';  
    cfg.trialdef.eventvalue = [201 202 203];  
    cfg.trialdef.prestim = 2.5;  
    cfg.trialdef.poststim = 2;  
    cfg  = ft_definetrial(cfg);   
    cfg.channel = {'MEGGRAD'};  
    cfg.demean = 'yes';    
    data{i,1}  = ft_preprocessing(cfg); 
end
%% 
cfg = [];
cfg.keepsampleinfo = 'no';
data_planar = ft_appenddata(cfg,data{:})
%% 
cfg = [];
cfg.viewmode='vertical';
cfg.continuous='no';
cfg.ylim=[ -1e-11  1e-11 ];
cfg.channel = {'MEG01*','MEG05*'};  
ft_databrowser(cfg, data_planar);
%%
cfg = [];
cfg.viewmode='vertical';
cfg.continuous='no';
ft_databrowser(cfg, data_planar);
%% 
data_planar.trialinfo(1:end,2)=1:length(data_planar.trialinfo)  
%%
cfg=[];
cfg.method  = 'summary';
%cfg.layout  = 'neuromag306planar.lay';
cfg.layout = 'neuromag306.lay'
planar_rjv1  = ft_rejectvisual(cfg, data_planar);
%%
trl_keep=planar_rjv1.trialinfo(1:end,2);
planar_rjv1.trialinfo(:,2)=[];
chan_rej=setdiff(data_planar.label,planar_rjv1.label);
chan_keep=data_planar.label;
chan_keep(find(ismember(data_planar.label(:,1), chan_rej(:,1))))= [];
%%
save([data_path,'chan_keep.mat'], 'chan_keep')
save([data_path,'trl_keep.mat'], 'trl_keep')

%% ICA Tutorial
trl=[data_path,'trl_keep.mat']; 
load(trl)
chan=[data_path,'chan_keep.mat']; 
load(chan)
cfg=[];
cfg.trials=trl_keep;
cfg.channel=chan_keep;
planar_rjv1=ft_selectdata(cfg,data_planar);
%%
cfg=[]; 
cfg.resamplefs = 250;   
planar_resamp = ft_resampledata(cfg, planar_rjv1);
%%
cfg=[];
cfg.hpfreq=1;
cfg.lpfreq=40;
cfg.padding=10.5;
cfg.padtype='zero';
planar_filt=ft_preprocessing(cfg, planar_resamp);
%% ICA Step 1 - ICA Decomposition
data_rank= rank(planar_filt.trial{1}*planar_filt.trial{1}');
%%
[u,s,v] = svd(planar_filt.trial{1}*planar_filt.trial{1}');
plot(log10(diag(s)),'-r*');
%%
cfg        = [];  
cfg.method = 'runica';  
cfg.numcomponent = data_rank;
planar_comp = ft_componentanalysis(cfg,planar_filt);   
%% ICA step 2 - Identifying components
cfg = [];
cfg.channel = [1:15]; 
cfg.continuous='no';
cfg.viewmode = 'component'; 
cfg.layout = 'neuromag306planar.lay';
%cfg.layout = 'neuromag306planar.lay';
ft_databrowser(cfg, planar_comp);
%% ICA Step 3 - Rejecting components
cfg = [];  
cfg.component = [1 3 4 10 21 22 23 24 27 29 36 39];  
planar_ica = ft_rejectcomponent(cfg, planar_comp, planar_rjv1);

%% 
figure
subplot(2,1,1)
plot((1:length(planar_rjv1.trial{1,200}(63,:)))/1000,planar_rjv1.trial{1,200}(63,:),(1:length(planar_ica.trial{1,200}(63,:)))/1000,planar_ica.trial{1,200}(63,:))
xlabel('time (s)')
title('MEG 1412 before (blue) and after (orange) ICA')

%% Further visual artefact rejection
planar_ica.trialinfo(1:end,2)=1:length(planar_ica.trialinfo)  
%% 
cfg = [];
cfg.viewmode='vertical';
cfg.continuous='no';
artf=ft_databrowser(cfg, planar_ica);
%%
planar_clean=ft_rejectartifact(artf,planar_ica)
%%
trl_keep2=planar_clean.trialinfo(1:end,2);
%%
save([data_path,'trl_keep2.mat'],'trl_keep2')
%%
save([data_path,'planar_comp.mat'], 'planar_comp')
%% %%%%%%%%%%%%%% Flux 8 %%%%%%%%%%%%%%%%%%%%%%
% data path
data_path = '/home/sanjeev/Desktop/BIO/MEG/sub-CA103/ses-1/meg/raw_sss/';
data=[data_path,'trl_keep2.mat']; 
load(data)
%%
cfg = [];
cfg.lpfilter = 'yes';
cfg.lpfreq = 30;      
cfg.padding   = 8;          
cfg.padtype  = 'zero';   
cfg.demean  = 'yes';    
cfg.detrend = 'no'; 
cfg.baselinewindow  = [-0.1 0];
planar_clean_filt = ft_preprocessing(cfg, planar_clean);
%% Select trials of specific conditions
idx=(find(planar_clean_filt.trialinfo(:,1)==201))';
cfg=[];
cfg.trials=idx;
data_task_relevant_target=ft_selectdata(cfg,planar_clean_filt);
%% 
clear idx
idx=(find(planar_clean_filt.trialinfo(:,1)==202))';
cfg=[];
cfg.trials=idx;
data_task_relevant_non_target=ft_selectdata(cfg,planar_clean_filt);
%% 
clear idx
idx=(find(planar_clean_filt.trialinfo(:,1)==203))';
cfg=[];
cfg.trials=idx;
data_task_irrelevant=ft_selectdata(cfg,planar_clean_filt);
%% Averaging the trial data
cfg = [];
data_erf_task_relevant_target = ft_timelockanalysis(cfg, data_task_relevant_target);
%% 
data_erf_task_relevant_non_target = ft_timelockanalysis(cfg, data_task_relevant_non_target);
%%
data_erf_task_irrelevant = ft_timelockanalysis(cfg, data_task_irrelevant);
%%
data_erf_task_relevant_target
%%
cfg = [];
cfg.layout       = 'neuromag306planar.lay';
cfg.baseline = [-0.1 0]; 
cfg.xlim = [-0.1 0.4]; 
cfg.ylim = [-3e-12 2e-11]; 
ft_multiplotER(cfg, data_erf_task_relevant_target,data_erf_task_relevant_non_target);
%%
cfg = [];
cfg.layout       = 'neuromag306planar.lay';
cfg.baseline = [-0.1 0]; 
cfg.xlim = [-0.1 0.4]; 
cfg.ylim = [-3e-12 2e-11]; 
ft_multiplotER(cfg, data_erf_task_relevant_target,data_erf_task_irrelevant);
%% 
cfg = [];
cfg.method = 'sum';  
data_erf_comb_task_relevant_target = ft_combineplanar(cfg,data_erf_task_relevant_target);

%%
cfg = [];
cfg.layout       = 'neuromag306planar.lay';
cfg.baseline = [-0.1 0];  
cfg.xlim = [0.110 0.110]; 
cfg.zlim = [0 8e-12];
ft_topoplotER(cfg, data_erf_task_relevant_target);
colorbar
%%
cfg = [];
cfg.layout       = 'neuromag306planar.lay';
cfg.baseline = [-0.1 0];  
cfg.xlim = [0.110 0.110]; 
cfg.zlim = [0 8e-12];
ft_topoplotER(cfg, data_erf_task_relevant_non_target);
colorbar
%%
cfg = [];
cfg.layout       = 'neuromag306planar.lay';
cfg.baseline = [-0.1 0];  
cfg.xlim = [0.110 0.110]; 
cfg.zlim = [0 8e-12];
ft_topoplotER(cfg, data_erf_task_irrelevant);
%%
colorbar
%% Time-frequency representations of power
cfg = [];
cfg.output = 'pow';
cfg.channel = 'MEGGRAD';
cfg.taper = 'hanning';
cfg.method = 'mtmconvol';
cfg.foi          = 2:2:30;
numfoi = length(cfg.foi);
cfg.t_ftimwin    =  ones(length(cfg.foi),1).* 0.5;
cfg.toi          = [-1.8 : 0.05: 1];
cfg.keeptrials = 'no'; 
tfr_low_data_task_relevant_target = ft_freqanalysis(cfg, data_task_relevant_target);
%%
tfr_low_data_task_relevant_non_target = ft_freqanalysis(cfg, data_task_relevant_non_target);
%%
tfr_low_data_task_irrelevant = ft_freqanalysis(cfg, data_task_irrelevant);
%%
cfg = [];
cfg.method = 'sum';
cfg.layout = 'neuromag306.lay';
tfr_low_left_c = ft_combineplanar(cfg,tfr_low_data_task_relevant_target);
%%
% Inspect the data structure of tfr_low_left
disp(tfr_low_data_task_relevant_target)
%% Plotting the TFR of the low frequency results
cfg = [];
cfg.baseline     = 'no'; 
cfg.showlabels   = 'no';	
cfg.xlim         = [-0.5 1];
cfg.zlim         = [-5e-24 5e-24] ;
cfg.channel = 'MEG2112';
ft_singleplotTFR(cfg, tfr_low_data_task_relevant_target);
%%
cfg = [];
cfg.baseline     = [-0.5 -0.25]; 
cfg.baselinetype = 'relative';
cfg.showlabels   = 'no';	
cfg.xlim         = [-0.5 1];      
cfg.zlim         = [0.4 1.7] ;	
cfg.channel = 'MEG2112';
ft_singleplotTFR(cfg, tfr_low_data_task_relevant_target);
%%
cfg = [];
cfg.baseline     = [-0.5 -0.3]; 
cfg.baselinetype = 'relative';
cfg.xlim=[-0.5 1];
cfg.zlim         = [0.4 2] ;	        
cfg.layout = 'neuromag306planar.lay';
ft_multiplotTFR(cfg, tfr_low_data_task_relevant_target);
title('TFR of power <30 Hz')
%% 
cfg.layout = 'neuromag306cmb.lay';
ft_multiplotTFR(cfg, tfr_low_left_c);
%%  Tutorial Constructing the forward model
mri_data_path = '/home/sanjeev/Desktop/BIO/MEG/sub-CA103/ses-1/anat/';

%%
megfile=[data_path,'sub-01_ses-01_task-dur_run-01_raw_sss.fif'];
mrifile=[mri_data_path,'sub-CA103_ses-1_T1w.nii'];
%%
mri = ft_read_mri(mrifile)
%%
ft_sourceplot([], mri);
%%
mri_reslice=ft_volumereslice([],mri)
%%
ft_sourceplot([], mri_reslice);
%%
headshape=ft_read_headshape(megfile);
%%

cfg = [];
cfg.method = 'fiducial';
cfg.coordsys = 'neuromag';
cfg.fiducial.nas    = [104 216 131];% position of nasion
cfg.fiducial.lpa    = [181 111 105];% position of LPA
cfg.fiducial.rpa    = [28 110 105];% position of RPA
mri_realigned_2 = ft_volumerealign(cfg, mri);
cfg = [];
mri_resliced_2 = ft_volumereslice(cfg, mri_realigned_2);
cfg = [];
ft_sourceplot(cfg, mri_resliced_2);

%%
grad=ft_read_sens(megfile,'senstype','meg');
%%
figure;
ft_plot_headshape(headshape);
ft_plot_sens(grad);
%%
%cfg=[];
%cfg.viewresult='yes';
%cfg.method='fiducial';
%cfg.coordsys='neuromag';
%cfg.fiducial.nas    = [120.53590824824978 123.53357571537607 211.50560209871685];% position of nasion
%cfg.fiducial.lpa    = [190.87994962432322 148.19031901286527 125.2312742330843];% position of LPA
%cfg.fiducial.rpa    = [66.11673437730116 153.42221220245077 116.27824487777482];% position of RPA
%mri_realigned2 = ft_volumerealign(cfg,mri_reslice)
%%
ft_sourceplot([], mri_realigned_2);
%%
headshape=ft_convert_units(headshape,'cm');
mri_realigned1=ft_convert_units(mri_resliced_2,'cm');
%%
nas = ft_warp_apply(mri_realigned1.transform, [mri_realigned1.cfg.previous.fiducial.nas]);
lpa = ft_warp_apply(mri_realigned1.transform, [mri_realigned1.cfg.previous.fiducial.lpa]);
rpa = ft_warp_apply(mri_realigned1.transform, [mri_realigned1.cfg.previous.fiducial.rpa]);
%%

%%
ft_determine_coordsys(mri_realigned1,'interactive','no')

%%
hold on;
ft_plot_headshape(headshape);
plot3(nas(1,1), nas(1,2), nas(1,3), 'm*');
plot3(lpa(1,1), lpa(1,2), lpa(1,3), 'm*');
plot3(rpa(1,1), rpa(1,2), rpa(1,3), 'm*');
%%
cfg=[];
cfg.method='headshape';
cfg.headshape.headshape=headshape;
cfg.headshape.icp='yes';
cfg.coordsys='neuromag';
mri_realigned3 =ft_volumerealign(cfg,mri_realigned1);
%%
mri_resliced_2 = ft_convert_units(mri_resliced_2,'cm');
cfg=[];
cfg.output = {'brain'};
mri_segm=ft_volumesegment(cfg,mri_resliced_2)
%%
cfg = [];
cfg.funparameter = 'brain';
ft_sourceplot(cfg, mri_segm);
%% preparing head model
cfg=[];
cfg.grad=grad;
cfg.method='singleshell';
headmodel=ft_prepare_headmodel(cfg,mri_segm);
%%
headmodel=ft_convert_units(headmodel,'cm');
%%
figure
ft_plot_sens(grad);
ft_plot_headshape(headshape);
ft_plot_headmodel(headmodel);
%%
cfg=[];
cfg.grad=grad;
cfg.headmodel=headmodel;
cfg.senstype='MEG';
cfg.grid.resolution=0.5;
cfg.grid.unit='cm';
cfg.normalize='yes';
cfg.channel=planar_clean.label;
grid=ft_prepare_leadfield(cfg);
%%
figure
ft_plot_sens(grad, 'style', '*b');
ft_plot_headmodel(headmodel, 'edgecolor', 'none'); alpha 0.4;
ft_plot_mesh(grid.pos(grid.inside,:));
%%
save([data_path,'headmodel_FLUX.mat'], 'headmodel')
save([data_path,'grid_FLUX.mat'], 'grid')
save([data_path,'mri_realign2_FLUX.mat'], 'mri_realigned_2')
save([data_path,'grad_FLUX.mat'], 'grad')
%% last file 
load([data_path,'headmodel_FLUX.mat'])
load([data_path,'grid_FLUX.mat'])
load([data_path,'mri_realign2_FLUX.mat'])
load([data_path,'grad_FLUX.mat'])

%% Source modeling of modulations of alpha band activty
cfg=[];
cfg.latency=[0.3 0.8];
data_task_relevant_target_sh=ft_selectdata(cfg,data_task_relevant_target);
%%
cfg=[];
cfg.latency=[0.3 0.8];
data_task_relevant_non_target_sh=ft_selectdata(cfg,data_task_relevant_non_target);
%%
cfg=[];
cfg.latency=[0.3 0.8];
data_task_irrelevant_sh=ft_selectdata(cfg,data_task_irrelevant);
%% Select also a baseline time-window of each condition.
cfg=[];
cfg.latency=[-0.8 -0.3];
data_task_relevant_target_sh_bsl=ft_selectdata(cfg,data_task_relevant_target);
%%
cfg.latency=[-0.8 -0.3];
data_task_relevant_non_target_sh_bsl=ft_selectdata(cfg,data_task_relevant_non_target);
%%
cfg.latency=[-0.8 -0.3];
data_task_irrelevant_sh_bsl=ft_selectdata(cfg,data_task_irrelevant);
%%
data_sh_cmb=ft_appenddata([], data_task_relevant_target_sh,data_task_relevant_non_target_sh,data_task_irrelevant_sh);
%%
data_sh_cmb_all=ft_appenddata([], data_task_relevant_target_sh, data_task_relevant_target_sh_bsl, data_task_relevant_non_target_sh, data_task_relevant_non_target_sh_bsl, data_task_irrelevant_sh,data_task_irrelevant_sh_bsl);
%%
cfg = [];
cfg.output = 'powandcsd';
cfg.method='mtmfft';
cfg.taper = 'hanning';
cfg.foi          = 10;
freq_data_task_relevant_target_sh=ft_freqanalysis(cfg, data_task_relevant_target_sh);
%%
freq_data_task_relevant_non_target_sh=ft_freqanalysis(cfg, data_task_relevant_non_target_sh);
%%
freq_data_task_irrelevant_sh=ft_freqanalysis(cfg, data_task_irrelevant_sh);
%%
freq_sh_cmb=ft_freqanalysis(cfg,data_sh_cmb);
%%
freq_task_relevant_target_sh_bsl=ft_freqanalysis(cfg, data_task_relevant_target_sh_bsl);
%%
freq_task_relevant_non_target_sh_bsl=ft_freqanalysis(cfg, data_task_relevant_non_target_sh_bsl);
%%
freq_task_irrelevant_sh_bsl=ft_freqanalysis(cfg, data_task_irrelevant_sh_bsl);
%%
freq_sh_cmb_all=ft_freqanalysis(cfg,data_sh_cmb_all);
%% Derive the spatial filters for the alpha band
tmp_data = ft_checkdata(freq_sh_cmb, 'cmbstyle', 'full');
crss_rank=rank(tmp_data.crsspctrm);
%%
tmp_data = ft_checkdata(freq_sh_cmb_all, 'cmbstyle', 'full');
crss_rank_all=rank(tmp_data.crsspctrm);
%%
cfg=[];
cfg.method='dics';
cfg.grad=grad;
cfg.frequency=freq_sh_cmb.freq;
cfg.sourcemodel=grid;
cfg.headmodel=headmodel;
cfg.dics.projectnoise='yes';
cfg.dics.keepfilter   = 'yes';
cfg.dics.realfilter   = 'yes';
cfg.dics.fixedori = 'yes';
cfg.dics.kappa=crss_rank;
source_cmb=ft_sourceanalysis(cfg,freq_sh_cmb);
%%
cfg.sourcemodel.filter=source_cmb.avg.filter;
source_task_relevant_target=ft_sourceanalysis(cfg,freq_data_task_relevant_target_sh);
%%
source_task_relevant_non_target=ft_sourceanalysis(cfg,freq_data_task_relevant_target_sh);
%%
source_task_irrelevant=ft_sourceanalysis(cfg,freq_data_task_irrelevant_sh);
%% Calculate the relative power difference between 'task relevant target' , 'task relevant non target' and 'task irrelevant' trials on source level.

source_con=source_task_relevant_target;
source_con.avg.pow=((source_task_irrelevant.avg.pow + source_task_relevant_non_target.avg.pow)-source_task_relevant_target.avg.pow)./(source_task_relevant_target.avg.pow+source_task_relevant_non_target.avg.pow+source_task_irrelevant.avg.pow);
%%
cfg=[];
cfg.method='dics';
cfg.grad=grad;
cfg.frequency=freq_sh_cmb_all.freq;
cfg.sourcemodel=grid;
cfg.headmodel=headmodel;
cfg.dics.projectnoise='yes';
cfg.dics.keepfilter   = 'yes';
cfg.dics.realfilter   = 'yes';
cfg.dics.fixedori = 'yes';
cfg.dics.kappa=crss_rank_all;
source_cmb_all=ft_sourceanalysis(cfg,freq_sh_cmb_all);
%%
cfg.sourcemodel.filter=source_cmb_all.avg.filter;
source_task_relevant_target_all=ft_sourceanalysis(cfg,freq_data_task_relevant_target_sh);
%%
source_task_relevant_non_target_all=ft_sourceanalysis(cfg,freq_data_task_relevant_non_target_sh);
%%
source_task_irrelevant_all=ft_sourceanalysis(cfg,freq_data_task_irrelevant_sh);
%%
source_task_relevant_target_sh_bsl_all=ft_sourceanalysis(cfg,freq_task_relevant_target_sh_bsl);
%%
source_task_relevant_non_target_sh_bsl_all=ft_sourceanalysis(cfg,freq_task_relevant_non_target_sh_bsl);
%%
source_task_irrelevant_sh_bsl_all=ft_sourceanalysis(cfg,freq_task_irrelevant_sh_bsl);
%% Plot the alpha modulation
source_con_all=source_task_relevant_target_all;
source_con_all.avg.pow=((source_task_relevant_target_all.avg.pow+source_task_relevant_non_target_all.avg.pow+source_task_irrelevant_all.avg.pow)-(source_task_relevant_target_sh_bsl_all.avg.pow+source_task_relevant_non_target_sh_bsl_all.avg.pow+source_task_irrelevant_sh_bsl_all.avg.pow))./((source_task_relevant_target_all.avg.pow+source_task_relevant_non_target_all.avg.pow+source_task_irrelevant_all.avg.pow));
%%
cfg=[];
cfg.parameter='avg.pow';
cfg.interpmethod  ='nearest';
source_conint=ft_sourceinterpolate(cfg,source_con,mri_realigned_2);
%%
source_conint_all=ft_sourceinterpolate(cfg,source_con_all,mri_realigned_2);
%% Plot the results regarding the baseline comparison. 
cfg=[];
cfg.method='ortho';
cfg.funparameter='pow';
cfg.maskparameter=cfg.funparameter;
cfg.funcolorlim=[-0.5 0.5];
cfg.opacitylim=[-0.5 0.5];
[~,maxidx]=max(abs(source_conint_all.pow));
cfg.location = source_conint_all.pos(maxidx,:);
cfg.opacitymap='vdown';
ft_sourceplot(cfg,source_conint_all);
%%
cfg=[];
cfg.method='ortho';
cfg.funparameter='pow';
cfg.maskparameter=cfg.funparameter;
cfg.funcolorlim=[-0.3 0.3];
cfg.opacitylim=[-0.3 0.3];
[~,maxidx]=max(abs(source_conint.pow));
cfg.location = source_conint.pos(maxidx,:);
cfg.opacitymap='vdown';
ft_sourceplot(cfg,source_conint);
%%