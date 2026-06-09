function bp = bandpower_custom(signal, fs, flo, fhi)

[pxx,f] = pwelch(signal,[],[],[],fs);

idx = f >= flo & f <= fhi;

bp = trapz(f(idx), pxx(idx));

end