% % % indoor_data:  9 × 839
% % % outdoor_data: 9 × 839
% % 
% % x = linspace(0, 500, 839);
% % 
% % figure;
% % 
% % subplot(2,1,1);
% % plot_mean_sigma(x, indoor_data, [0 0.35 0.65], [0.35 0.65 0.85]);
% % title('(a) Indoor scenario.', 'Interpreter', 'latex');
% % 
% % subplot(2,1,2);
% % plot_mean_sigma(x, outdoor_data, [1 0.2 0.05], [1 0.55 0.45]);
% % title('(b) Outdoor (open area) scenario.', 'Interpreter', 'latex');
% 
% 
% function plot_mean_sigma_ht(x, data, lineColor, fillColor)
% 
%     mean_y = mean(data, 1);
%     std_y  = std(data, 0, 1);
% 
%     upper_y = mean_y + std_y;
%     lower_y = mean_y - std_y;
% 
%     hold on;
% 
%     fill([x, fliplr(x)], ...
%          [upper_y, fliplr(lower_y)], ...
%          fillColor, ...
%          'FaceAlpha', 0.45, ...
%          'EdgeColor', 'none');
% 
%     plot(x, mean_y, ...
%          'Color', lineColor, ...
%          'LineWidth', 0.9);
% 
%     grid on;
%     box on;
% 
%     % xlabel('subCarrier', 'FontName', 'Times New Roman');
%     xlabel('Delay (\mus)');
%     ylabel('|h(n)|');
%     xlim([0 5]);
% 
%     set(gca, ...
%         'FontName', 'Times New Roman', ...
%         'FontSize', 12, ...
%         'LineWidth', 1);
% 
%     % xlim([0 939]);
%     % axis tight
% 
% end

% indoor_data:  9 × 839
% outdoor_data: 9 × 839

% x = linspace(0, 5, 839);   % Delay in us
% 
% figure;
% 
% subplot(2,1,1);
% plot_mean_sigma_ht(x, indoor_data, [0 0.35 0.65], [0.35 0.65 0.85], true);
% title('(a) Indoor scenario.', 'Interpreter', 'latex');
% 
% subplot(2,1,2);
% plot_mean_sigma_ht(x, outdoor_data, [1 0.2 0.05], [1 0.55 0.45], true);
% title('(b) Outdoor (open area) scenario.', 'Interpreter', 'latex');


function plot_mean_sigma_ht(x, data, lineColor, fillColor, doInset)

    if nargin < 5
        doInset = false;
    end

    mean_y = mean(data, 1);
    std_y  = std(data, 0, 1);

    upper_y = mean_y + std_y;
    lower_y = mean_y - std_y;

    axMain = gca;
    hold(axMain, 'on');

    fill(axMain, [x, fliplr(x)], ...
         [upper_y, fliplr(lower_y)], ...
         fillColor, ...
         'FaceAlpha', 0.45, ...
         'EdgeColor', 'none');

    plot(axMain, x, mean_y, ...
         'Color', lineColor, ...
         'LineWidth', 0.9);

    grid(axMain, 'on');
    box(axMain, 'on');

    xlabel(axMain, 'Delay ($\mu$s)', 'Interpreter', 'latex');
    ylabel(axMain, '$|h(t)|$', 'Interpreter', 'latex');

    xlim(axMain, [0 5]);

    set(axMain, ...
        'FontName', 'Times New Roman', ...
        'FontSize', 12, ...
        'LineWidth', 1);

    if doInset
        mainPos = get(axMain, 'Position');

        insetPos = [
            mainPos(1) + 0.53 * mainPos(3), ...
            mainPos(2) + 0.50 * mainPos(4), ...
            0.36 * mainPos(3), ...
            0.38 * mainPos(4)
        ];

        axInset = axes('Position', insetPos);
        hold(axInset, 'on');

        fill(axInset, [x, fliplr(x)], ...
             [upper_y, fliplr(lower_y)], ...
             fillColor, ...
             'FaceAlpha', 0.45, ...
             'EdgeColor', 'none');

        plot(axInset, x, mean_y, ...
             'Color', lineColor, ...
             'LineWidth', 1.1);

        grid(axInset, 'on');
        box(axInset, 'on');

        xlim(axInset, [0 0.20]);

        idx = x >= 0 & x <= 0.20;
        yMin = min(lower_y(idx));
        yMax = max(upper_y(idx));
        pad = 0.08 * (yMax - yMin + eps);
        ylim(axInset, [yMin - pad, yMax + pad]);

        set(axInset, ...
            'FontName', 'Times New Roman', ...
            'FontSize', 9, ...
            'LineWidth', 0.9);

        xlabel(axInset, '');
        ylabel(axInset, '');

        title(axInset, 'Zoom', ...
            'FontName', 'Times New Roman', ...
            'FontSize', 9, ...
            'FontWeight', 'normal');

        % axes(axMain);
    end
end

