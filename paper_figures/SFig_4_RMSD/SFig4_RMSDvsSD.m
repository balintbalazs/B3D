%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare image noise level with B³D compression artifacts %
% Author: Balint Balazs                                    %
% contact: balint.balazs@embl.de                           %
% 26.06.2016                                               %
% EMBL Heidelberg, Cell Biology and Biophysics             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% define data location
clear variables;;
baseFolder = fileparts(matlab.desktop.editor.getActiveFilename);
fileName = '\drosophila_single_384x864x100_filtered.h5';

%%
v = H5.get_libversion;
%%
D1 = h5read([baseFolder, fileName], '/B3D_0.0');
D1 = double(D1);
%% load compressed images
D1C = h5read([baseFolder, fileName], '/B3D_1.0');
D1C = double(D1C);
%% calculate means
mD1 = mean(D1,3);
mD1C = mean(D1C,3);

%% calculate standard deviations (normalized by N) along 3rd dimension
sdD1 = std(D1,1,3);     % = sqrt(mean((repmat(mD1,[1,1,100]) - D1) .^ 2, 3));
sdD1C = std(D1C,1,3);

%% calculate mean squared error
MSE = mean((D1 - D1C) .^ 2, 3);

%% calculate root mean squared error
RMSE = sqrt(MSE);

mean(sdD1(:)) / mean(RMSE(:))

%%
figureSize = 500;
borderWidth = 1.5;
lineWidth = 2;
fontSize = 16;
scheme = 'RdYlBu';
cmap = flip(brewermap(256, scheme));
crange = [0,40];
%% plot standard deviation of uncompressed and RMSE of compressed
f1 = figure(1);
cla reset;
set(f1, 'Position', [50,250,1.5*figureSize,figureSize])
set(gcf, 'Color', 'w')

subplot(1,2,2)
imagesc(RMSE', crange)
colormap(cmap);
cbar1 = colorbar;
set(gca, 'FontSize', fontSize)
axis image
set(gca, 'LineWidth', lineWidth)
set(cbar1, 'LineWidth', lineWidth)
set(gca, 'xticklabel','', 'yticklabel', '', 'xtick', [], 'ytick', [])
set(cbar1, 'ytick', crange)
title('root mean square deviation')

subplot(1,2,1)
set(gca, 'LineWidth', lineWidth)
imagesc(sdD1', crange)
cbar2 = colorbar
set(gca, 'FontSize', fontSize)
set(gca, 'LineWidth', lineWidth)
set(cbar2, 'LineWidth', lineWidth)
set(gca, 'xticklabel','', 'yticklabel', '', 'xtick', [], 'ytick', [])
set(cbar2, 'ytick', crange)
axis image
title('standard deviation')

%% save figure
export_fig([baseFolder, '\SFig4_RMSDvsSD.pdf']);