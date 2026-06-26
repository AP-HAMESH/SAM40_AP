function fvec = extract_features(ep, fs)

% ep should be: Channels x Samples
% Example: 32 x 512

nCh = size(ep,1);

delta = zeros(1,nCh);
theta = zeros(1,nCh);
alpha = zeros(1,nCh);
beta  = zeros(1,nCh);
gamma = zeros(1,nCh);

for ch = 1:nCh

    x = ep(ch,:);

    delta(ch) = bandpower_custom(x,fs,0.5,4);
    theta(ch) = bandpower_custom(x,fs,4,8);
    alpha(ch) = bandpower_custom(x,fs,8,13);
    beta(ch)  = bandpower_custom(x,fs,13,30);
    gamma(ch) = bandpower_custom(x,fs,30,45);

end

eps_val = 1e-10;

theta_alpha = theta ./ (alpha + eps_val);
beta_alpha  = beta  ./ (alpha + eps_val);
engagement  = (theta + alpha) ./ (beta + eps_val);

mean_feat = mean(ep,2)';
std_feat  = std(ep,0,2)';
skew_feat = skewness(ep,0,2)';
kurt_feat = kurtosis(ep,0,2)';

stft_e = [];

for ch = 1:min(nCh,8)

    [~,~,~,P] = spectrogram(ep(ch,:),64,48,64,fs);

    stft_e(end+1) = mean(P(:));

end

stft_stats = [
    mean(stft_e)
    std(stft_e)
    max(stft_e)
    min(stft_e)
    sum(stft_e)
]';

fvec = [
    delta ...
    theta ...
    alpha ...
    beta ...
    gamma ...
    theta_alpha ...
    beta_alpha ...
    engagement ...
    mean_feat ...
    std_feat ...
    skew_feat ...
    kurt_feat ...
    stft_stats
];

fvec(~isfinite(fvec)) = 0;

end