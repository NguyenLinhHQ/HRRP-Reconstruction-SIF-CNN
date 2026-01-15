function mssim = compute_mssim_1d(x, y, window_len)
    if nargin < 3
        window_len = 11;  % Default window size
    end
    C1 = 1e-4;
    C2 = 9e-4;

    N = length(x);
    if N ~= length(y)
        error('Input vectors must have the same length');
    end

    n_windows = N - window_len + 1;
    ssim_vals = zeros(1, n_windows);

    for i = 1:n_windows
        x_win = x(i:i+window_len-1);
        y_win = y(i:i+window_len-1);

        mu_x = mean(x_win);
        mu_y = mean(y_win);
        sigma_x = std(x_win);
        sigma_y = std(y_win);
        sigma_xy = cov(x_win, y_win); sigma_xy = sigma_xy(1,2);

        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2);
        denominator = (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2);

        ssim_vals(i) = numerator / denominator;
    end

    mssim = mean(ssim_vals);
end
