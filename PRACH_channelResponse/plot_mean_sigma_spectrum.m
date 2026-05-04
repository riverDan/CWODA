% % indoor_data:  9 × 839
% % outdoor_data: 9 × 839
% 
% x = linspace(0, 500, 839);
% 
% figure;
% 
% subplot(2,1,1);
% plot_mean_sigma(x, indoor_data, [0 0.35 0.65], [0.35 0.65 0.85]);
% title('(a) Indoor scenario.', 'Interpreter', 'latex');
% 
% subplot(2,1,2);
% plot_mean_sigma(x, outdoor_data, [1 0.2 0.05], [1 0.55 0.45]);
% title('(b) Outdoor (open area) scenario.', 'Interpreter', 'latex');


function plot_mean_sigma_spectrum(x, data, lineColor, fillColor)

    mean_y = mean(data, 1);
    std_y  = std(data, 0, 1);

    upper_y = mean_y + std_y;
    lower_y = mean_y - std_y;

    hold on;

    fill([x, fliplr(x)], ...
         [upper_y, fliplr(lower_y)], ...
         fillColor, ...
         'FaceAlpha', 0.45, ...
         'EdgeColor', 'none');

    plot(x, mean_y, ...
         'Color', lineColor, ...
         'LineWidth', 0.9);

    grid on;
    box on;

    xlabel('Frequency (MHz)', 'FontName', 'Times New Roman');
    ylabel('Amplitude (dB)', 'FontName', 'Times New Roman');

    set(gca, ...
        'FontName', 'Times New Roman', ...
        'FontSize', 12, ...
        'LineWidth', 1);

    xlim([0 939]);
    axis tight

end
