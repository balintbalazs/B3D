%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Figure 1c for B³D compression paper                            %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 27.06.2017                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% define data
baseFolder = fileparts(matlab.desktop.editor.getActiveFilename);

labels = {'drosophila',...
'phallusia',...
'zebrafish',...
'simulation',...
'microtubules',...
'lifeact',...
'dapi',...
'membrane',...
'vsvg'};

% labels = {'SPIM-drosophila',...
% 'SPIM-phallusia',...
% 'SPIM-zebrafish',...
% 'SMLM-simulation',...
% 'SMLM-microtubules',...
% 'SMLM-lifeact',...
% 'screening-dapi',...
% 'screening-pm-647',...
% 'screening-vsvg-cfp'};

compressions = {'B3D lossless', 'B3D WNL'};

ratios = [7.523498676	24.634113;... % drosophila - DONE
18.4218 	44.0417;...   % phallusia - DONE
8.492567	26.759312;...   % zebrafish - DONE
1.573679	4.567811;... % simu - DONE swapped
1.435835    5.241922;... % MT ROI6 - DONE swapped
1.313104	5.075731;... % lifeact ROI8 - DONE swapped
22.395218	37.101418;... % dapi - DONE
8.278097	14.744775;... % pm647 - DONE
9.964772	18.321808];   % vsvg - DONE
%%
figureSize = 500;
borderWidth = 1.5;
lineWidth = 1.5;
scheme = 'RdYlBu';
numColumns = 9;

%% bar plot ratios
f5 = figure(5);
cla reset;
set(f5, 'Position', [50,550,figureSize,figureSize])
hold on
axis square
set(gcf, 'Color', 'w')
set(gca, 'LineWidth', 1)
h = gca;
h.YRuler.LineWidth = borderWidth;
h.XRuler.LineWidth = borderWidth;
h.GridColor = g(1);
grid on
set(gca, 'Box', 'on', 'Color', g(0.9), 'FontSize', 14)
colors = flip(brewermap(5, scheme));

barh(flip(ratios,2));
barh(flip(ratios,2));
colormap(colors);

ylim([0.5, numColumns+0.5]);
yticklabels(labels);
% ylim([0, 1000]);
% xlabel('compression speed (MB/s)');
xlabel('compression ratio');
title({'\fontsize{16}Comparing compression ratios'})
% set(gca, 'Xscale', 'log', 'Yscale', 'log') 
set(gca, 'YDir', 'Reverse')
leg3 = legend(flip(compressions));
set(leg3, 'Location', 'SouthEast');
set(leg3, 'Color', 'w', 'LineWidth', borderWidth);
%% save figure
export_fig([baseFolder, '\Fig1c_compressionBars.pdf'])


