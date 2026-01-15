%% =========================================================
%% This script reproduces the simulation results reported in:
%% Section III: Methodology
%% Section IV: Experimental Results
%% =========================================================
close all;
clear all;
clc;
%% I. Spectral Inverse Filtering (SIF)
%% Corresponds to Section III.B in the paper
%% Radar and signal parameters
pw = 1e-6;                         % Pulse width [s]
c = 3e8;                           % Speed of light [m/s]
duration = 4e-6;                   % Signal duration [s] about 600m
fs_list = [10, 20, 40, 60] / pw;   % Sampling frequencies [Hz]
snr_list = [-5, 0, 10, 20, 30, 40];% SNR values [dB]

%% Fighter jet target model
scatterer_ranges = 150 + [0, 8, 15, 20, 26, 30];   % [m] from radar
target_delays_s = 2 * scatterer_ranges / c;        % Round-trip delay [s]
target_amps = [0.8, 1.5, 2.0, 1.8, 1.2, 0.6];      % Reflector amplitudes
p_target = length(target_amps);

%% Part 1: Sampling Frequency Sweep
%% Reproduces figures analyzing the effect of fs on HRRP resolution
figure('Name','SIF vs Sampling Rate (Clean Real)', ...
       'Position',[50 50 1400 1000]);   

tlo = tiledlayout(2,2, ...
        'TileSpacing','compact', ...   
        'Padding','compact');          

title(tlo,'Effect of Sampling Rate on HRRP', ...
       'FontName','Times New Roman', ...
       'FontSize',18, ...
       'FontWeight','bold');

for k = 1:length(fs_list)
    fs = fs_list(k);
    dt = 1/fs;
    t = 0:dt:duration;
    N = length(t);
    p = round(pw * fs);

    % Real transmit pulse
    pulse = [ones(1, p), zeros(1, N - p)];
    pulse = pulse / norm(pulse);
    H = fft(pulse); H(abs(H) < 1e-6) = 1e-6;
    K = ifft(1 ./ H);

    % Clean signal (real)
    sig_clean = zeros(1, N);
    for j = 1:p_target
        idx = round(target_delays_s(j) * fs);
        if idx + p - 1 <= N
            sig_clean(idx:idx+p-1) = sig_clean(idx:idx+p-1) + target_amps(j);
        end
    end

    % SIF Output
    sif_out = abs(ifft(fft(sig_clean) .* fft(K)));
    sif_out = sif_out / max(sif_out);
    range_axis = (0:N-1) * dt * c / 2;

    nexttile;
    plot(range_axis, sif_out, 'r', 'LineWidth', 1.2); hold on;
    plot(range_axis, abs(sig_clean)/max(abs(sig_clean)), 'k--');
    for j = 1:p_target
        x = scatterer_ranges(j);
        y = target_amps(j) / max(target_amps);
        text(x, y + 0.05, sprintf('%d', j), 'FontSize', 14, 'Color', 'b', 'FontName','Times New Roman');
    end
    title(sprintf('fs = %d MHz', fs/1e6), 'FontName','Times New Roman', 'FontSize', 16);
    xlabel('Range (m)', 'FontName','Times New Roman', 'FontSize', 16);
    ylabel('Amplitude', 'FontName','Times New Roman', 'FontSize', 16);
    grid on; xlim([50 400]); ylim([0 1.2]);
    set(gca,'FontName','Times New Roman','FontSize',16);
end


%%  Part 2: SNR Sweep at fs = 60 MHz 
%% Corresponds to noisy SIF results in Section IV
fs = 60e6; 
dt = 1/fs;
t = 0:dt:duration;
N = length(t);
p = round(pw * fs);

% Transmit pulse and SIF kernel
pulse = [ones(1, p), zeros(1, N - p)];
pulse = pulse / norm(pulse);
H = fft(pulse); H(abs(H) < 1e-6) = 1e-6;
K = ifft(1 ./ H);

% Generate clean signal (fixed)
sig_clean = zeros(1, N);
for j = 1:p_target
    idx = round(target_delays_s(j) * fs);
    if idx + p - 1 <= N
        sig_clean(idx:idx+p-1) = sig_clean(idx:idx+p-1) + target_amps(j);
    end
end
signal_power = mean(sig_clean.^2);
sig_clean_norm = abs(sig_clean) / max(abs(sig_clean));
range_axis = (0:N-1) * dt * c / 2;

figure('Name','SNR Sweep at fs = 60MHz','Position',[100 100 1200 800]);

tlo = tiledlayout(3,2, 'TileSpacing','compact', 'Padding','compact');  % 3x2 grid, no spacing
title(tlo, 'Effect of SNR on HRRP (fs = 60 MHz)', 'FontName','Times New Roman','FontSize', 14, 'FontWeight', 'bold');

for i = 1:length(snr_list)
    SNR_dB = snr_list(i);
    snr_linear = 10^(SNR_dB / 10);
    noise_power = signal_power / snr_linear;
    noise = sqrt(noise_power/2) * (randn(1, N)); % + 1j * randn(1, N));
    noisy_sig = sig_clean + noise;

    % Apply SIF
    sif_out = abs(real(ifft(fft(noisy_sig) .* fft(K))));
    sif_out = sif_out / max(sif_out);

    nexttile;
    plot(range_axis, sif_out, 'r', 'LineWidth', 1.2); hold on;
    plot(range_axis, sig_clean_norm, 'k--', 'LineWidth', 1);
    for j = 1:p_target
        x = scatterer_ranges(j);
        y = target_amps(j) / max(target_amps);
        text(x, y + 0.05, sprintf('%d', j), 'FontName','Times New Roman','FontSize', 14, 'Color', 'b');
    end
    title(sprintf('SNR = %d dB', SNR_dB), 'FontName','Times New Roman','FontSize', 14);
    legend('SIF Output (Noisy)', 'Clean Echoes', 'Location', 'northeast', 'FontName','Times New Roman','FontSize', 14);
    xlabel('Range [m]'); ylabel('Amplitude');
    grid on; xlim([50 400]); ylim([0 1.2]);
    set(gca, 'FontName','Times New Roman','FontSize', 14);
end


%% II CNN%% =========================================================
%% II. Dataset generation for CNN training
%% Corresponds to Section III.C
%% =========================================================
% Parameters
pw = 1e-6;                         % Pulse width [s]
duration = 4e-6;                   % Duration 600m
fs = 60e6; 
dt = 1/fs;
t = 0:dt:duration;
p = round(pw * fs);               % Pulse length in samples => 60
Nsamples = length(t);             % 
nSignals = 200000;
SNR_range = -5:1:30;              % Diverse SNRs for training

%% Generate transmit pulse and SIF kernel
pulse = [ones(1, p), zeros(1, Nsamples - p)];
pulse = pulse / norm(pulse);     % Normalize
H = fft(pulse);
H(abs(H) < 1e-6) = 1e-6;
K = ifft(1 ./ H);                 % SIF kernel

%% Generate dataset
X = zeros(nSignals, Nsamples);   % Noisy signals (input)
Y = zeros(nSignals, Nsamples);   % Clean SIF outputs (label)

for i = 1:nSignals
    SNR_dB = SNR_range(randi(length(SNR_range))); % Random SNR

    % Random reflectors
    n_reflectors = randi([6 8]);
    delays = sort(randi([10 Nsamples - p - 10], 1, n_reflectors));
    amps = rand(1, n_reflectors) * 1.5 + 0.5;

    % Clean signal
    sig_clean = zeros(1, Nsamples);
    for j = 1:n_reflectors
        idx = delays(j);
        sig_clean(idx:idx+p-1) = sig_clean(idx:idx+p-1) + amps(j);
    end

    % Ideal SIF output
    SIF_clean = real(ifft(fft(sig_clean) .* fft(K)));
    SIF_clean = SIF_clean / (max(abs(SIF_clean)) + eps);

    % Add noise
    snr_linear = 10^(SNR_dB / 10);
    signal_power = sum(sig_clean.^2) / length(sig_clean);
    noise_power = signal_power / snr_linear;
    noise = sqrt(noise_power) * randn(1, length(sig_clean));
    noisy_sig = sig_clean + noise;
    noisy_sig = noisy_sig / (max(abs(noisy_sig)) + eps);

    % Store
    X(i, :) = noisy_sig;
    Y(i, :) = SIF_clean;
end

%% Reshape for CNN
XTrain = reshape(X', [Nsamples, 1, 1, nSignals]);
YTrain = reshape(Y', [Nsamples, 1, 1, nSignals]);

%% Define CNN
%% CNN architecture for HRRP reconstruction
%% Corresponds to Section III.D and Figure 2
layers = [
    imageInputLayer([Nsamples 1 1], 'Normalization', 'none')
    convolution2dLayer([5 1], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([9 1], 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([15 1], 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 1], 1, 'Padding', 'same')
    regressionLayer
];
analyzeNetwork(layers);
options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% Train CNN
net = trainNetwork(XTrain, YTrain, layers, options);
save('trainedNet.mat', 'net');

%% Evaluation
%% =========================================================
%% Monte Carlo evaluation over multiple SNR levels
%% Metrics: RMSE, PSNR, MSSIM (Section IV-C)
%% =========================================================
snr_range = -5:5:40;
num_trials = 100;
avg_rmse_cnn = zeros(size(snr_range));
avg_rmse_sif = zeros(size(snr_range));
avg_psnr_cnn = zeros(size(snr_range));
avg_psnr_sif = zeros(size(snr_range));
avg_mssim_cnn = zeros(size(snr_range));
avg_mssim_sif = zeros(size(snr_range));


for s = 1:length(snr_range)
    SNR_test = snr_range(s);

    total_rmse_cnn = 0; total_rmse_sif = 0;
    total_psnr_cnn = 0; total_psnr_sif = 0;
    total_mssim_cnn = 0; total_mssim_sif = 0;
   

    for i = 1:num_trials
        n_reflectors = randi([6 8]);
        delays = sort(randi([10 Nsamples - p - 10], 1, n_reflectors));
        amps = rand(1, n_reflectors) * 1.5 + 0.5;
        sig_clean = zeros(1, Nsamples);
        for j = 1:n_reflectors
            idx = delays(j);
            sig_clean(idx:idx+p-1) = sig_clean(idx:idx+p-1) + amps(j);
        end

        SIF_clean = real(ifft(fft(sig_clean) .* fft(K)));
        SIF_clean = SIF_clean / (max(abs(SIF_clean)) + eps);

        snr_linear = 10^(SNR_test / 10);
        signal_power = sum(sig_clean.^2) / length(sig_clean);
        noise_power = signal_power / snr_linear;
        noise = sqrt(noise_power) * randn(1, length(sig_clean));
        noisy_sig = sig_clean + noise;
        noisy_sig = noisy_sig / (max(abs(noisy_sig)) + eps);

        testInput = reshape(noisy_sig', [Nsamples, 1, 1]);
        cnn_out = predict(net, testInput);
        cnn_out = reshape(cnn_out, 1, []);
        cnn_out = cnn_out / (max(abs(cnn_out)) + eps);

        sif_out = abs(real(ifft(fft(noisy_sig) .* fft(K))));
        sif_out = sif_out / (max(abs(sif_out)) + eps);

        rmse_cnn = sqrt(mean((cnn_out - SIF_clean).^2));
        rmse_sif = sqrt(mean((sif_out - SIF_clean).^2));
        psnr_cnn = 20 * log10(max(SIF_clean) / (rmse_cnn + eps));
        psnr_sif = 20 * log10(max(SIF_clean) / (rmse_sif + eps));
        mssim_cnn = compute_mssim_1d(cnn_out, SIF_clean);
        mssim_sif = compute_mssim_1d(sif_out, SIF_clean);

        total_rmse_cnn = total_rmse_cnn + rmse_cnn;
        total_rmse_sif = total_rmse_sif + rmse_sif;
        total_psnr_cnn = total_psnr_cnn + psnr_cnn;
        total_psnr_sif = total_psnr_sif + psnr_sif;
        total_mssim_cnn = total_mssim_cnn + mssim_cnn;
        total_mssim_sif = total_mssim_sif + mssim_sif;
       
        if i == 10
        if ismember(SNR_test, [-5 0 10 20 30 40])
    % Plot one example
    figure;
    plot(SIF_clean/max(SIF_clean), 'g--', 'LineWidth', 1.5); hold on;
    plot(sif_out/max(sif_out), 'r', 'LineWidth', 1.2);
    plot(cnn_out/max(cnn_out), 'b', 'LineWidth', 1.2);
    plot(sig_clean/max(sig_clean), 'k:','LineWidth', 1);
    plot(noisy_sig, 'm:','LineWidth', 1);
    legend('Ground Truth (SIF clean)', 'SIF (noisy)', 'CNN','Clean signal','Noisy signal');
    title(sprintf('Example HRRP at %d dB SNR', SNR_test));
    xlabel('Range bins'); ylabel('Amplitude'); grid on; ylim([-0.15 1.5])
    set(gca, 'FontName','Times New Roman','FontSize', 10);   
        end
        end
    end

    avg_rmse_cnn(s) = total_rmse_cnn / num_trials;
    avg_rmse_sif(s) = total_rmse_sif / num_trials;
    avg_psnr_cnn(s) = total_psnr_cnn / num_trials;
    avg_psnr_sif(s) = total_psnr_sif / num_trials;
    avg_mssim_cnn(s) = total_mssim_cnn / num_trials;
    avg_mssim_sif(s) = total_mssim_sif / num_trials;
    
    fprintf('\n--- SNR = %d dB ---\n', SNR_test);
    fprintf('CNN:  RMSE=%.4f | PSNR=%.2f | MSSIM=%.4f ', ...
        avg_rmse_cnn(s), avg_psnr_cnn(s), avg_mssim_cnn(s));
    fprintf('SIF:  RMSE=%.4f | PSNR=%.2f | MSSIM=%.4f', ...
        avg_rmse_sif(s), avg_psnr_sif(s), avg_mssim_sif(s));
   
end

figure; plot(snr_range, avg_rmse_cnn, 'b-o', snr_range, avg_rmse_sif, 'r-s');
title('RMSE of CNN and SIF outputs across SNR levels'); xlabel('SNR (dB)'); ylabel('RMSE'); legend('CNN', 'SIF'); grid on;
set(gca, 'FontName','Times New Roman','FontSize', 10);   
figure; plot(snr_range, avg_psnr_cnn, 'b-o', snr_range, avg_psnr_sif, 'r-s');
title('PSNR of CNN and SIF outputs across SNR levels'); xlabel('SNR (dB)'); ylabel('PSNR (dB)'); legend('CNN', 'SIF', 'Location', 'northwest'); grid on;
set(gca, 'FontName','Times New Roman','FontSize', 10);   
figure; plot(snr_range, avg_mssim_cnn, 'b-o', snr_range, avg_mssim_sif, 'r-s');
title('MSSIM of CNN and SIF outputs across SNR levels'); xlabel('SNR (dB)'); ylabel('MSSIM'); legend('CNN', 'SIF', 'Location', 'northwest'); grid on;
set(gca, 'FontName','Times New Roman','FontSize', 10);   

