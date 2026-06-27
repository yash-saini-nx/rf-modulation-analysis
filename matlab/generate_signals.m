% RF Modulation Analysis - MATLAB Signal Generator
%
% Creates synthetic AM, FM, SSB, BPSK, QPSK, QAM16, and QAM64 IQ signals.
% Noise is added with AWGN at SNR values from -10 dB to +20 dB.
% Output: ../data/signals.mat with iqData, labels, snrs, modTypes, snrValues.

clear; clc; close all;
rng(42);

outputFile = fullfile('..', 'data', 'signals.mat');
if ~exist(fullfile('..', 'data'), 'dir')
    mkdir(fullfile('..', 'data'));
end

modTypes = {'AM', 'FM', 'SSB', 'BPSK', 'QPSK', 'QAM16', 'QAM64'};
snrValues = -10:2:20;
samplesPerClassPerSnr = 120;
segmentLength = 128;
sampleRate = 2000;
carrierFreq = 200;

totalSamples = numel(modTypes) * numel(snrValues) * samplesPerClassPerSnr;
iqData = zeros(totalSamples, 2, segmentLength);
labels = strings(totalSamples, 1);
snrs = zeros(totalSamples, 1);
row = 1;

for m = 1:numel(modTypes)
    modName = modTypes{m};
    for s = 1:numel(snrValues)
        snrDb = snrValues(s);
        for n = 1:samplesPerClassPerSnr
            cleanSignal = make_modulated_signal(modName, segmentLength, sampleRate, carrierFreq);
            noisySignal = add_awgn(cleanSignal, snrDb);
            noisySignal = noisySignal ./ max(abs(noisySignal) + eps);
            iqData(row, 1, :) = real(noisySignal);
            iqData(row, 2, :) = imag(noisySignal);
            labels(row) = modName;
            snrs(row) = snrDb;
            row = row + 1;
        end
    end
end

save(outputFile, 'iqData', 'labels', 'snrs', 'modTypes', 'snrValues');
fprintf('Saved %d IQ examples to %s\n', totalSamples, outputFile);

function x = make_modulated_signal(modName, N, fs, fc)
    t = (0:N-1) / fs;
    phaseOffset = 2 * pi * rand();
    switch modName
        case 'AM'
            message = 0.6 * sin(2*pi*30*t + phaseOffset);
            x = (1 + message) .* exp(1j * 2*pi*fc*t);
        case 'FM'
            message = sin(2*pi*25*t + phaseOffset);
            freqDev = 80;
            phase = 2*pi*fc*t + 2*pi*freqDev*cumsum(message)/fs;
            x = exp(1j * phase);
        case 'SSB'
            message = sin(2*pi*35*t + phaseOffset) + 0.5*sin(2*pi*55*t);
            analyticMessage = hilbert(message);
            x = analyticMessage .* exp(1j * 2*pi*fc*t);
        case 'BPSK'
            bits = randi([0 1], 1, ceil(N/8));
            symbols = 2*bits - 1;
            x = repelem(symbols, 8);
            x = x(1:N) .* exp(1j * phaseOffset);
        case 'QPSK'
            sym = randi([0 3], 1, ceil(N/8));
            constellation = exp(1j * (pi/4 + sym*pi/2));
            x = repelem(constellation, 8);
            x = x(1:N) .* exp(1j * phaseOffset);
        case 'QAM16'
            x = make_qam_signal(16, N, phaseOffset);
        case 'QAM64'
            x = make_qam_signal(64, N, phaseOffset);
        otherwise
            error('Unknown modulation type: %s', modName);
    end
end

function x = make_qam_signal(M, N, phaseOffset)
    side = sqrt(M);
    levels = -(side-1):2:(side-1);
    i = levels(randi(numel(levels), 1, ceil(N/8)));
    q = levels(randi(numel(levels), 1, ceil(N/8)));
    symbols = i + 1j*q;
    symbols = symbols ./ sqrt(mean(abs(symbols).^2));
    x = repelem(symbols, 8);
    x = x(1:N) .* exp(1j * phaseOffset);
end

function y = add_awgn(x, snrDb)
    signalPower = mean(abs(x).^2);
    noisePower = signalPower / (10^(snrDb/10));
    noise = sqrt(noisePower/2) * (randn(size(x)) + 1j*randn(size(x)));
    y = x + noise;
end
