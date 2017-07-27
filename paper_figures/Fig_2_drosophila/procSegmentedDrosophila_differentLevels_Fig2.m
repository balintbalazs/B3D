%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare segmentation of original and B³D compressed images              %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 23.06.2017                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% compare drosophila nuclei segmentation
clear variables;
%%
baseFolder = 'D:\GPU_compression\!paper_figures\Fig_2_drosophila\';
saveFolder = [fileparts(matlab.desktop.editor.getActiveFilename), '\'];
compressionLevels = [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5];
fileNameFormat = 'drosophila_1_masked24_1024x2048x211_filtered_B3D_%.2f_even_smaller_Probabilities.h5';
labels = {...
    'B3D_lossless', 'B3D_0.25', 'B3D_0.5', 'B3D_1.0', 'B3D_1.5', 'B3D_2.0', 'B3D_2.5', 'B3D_3.0', 'B3D_4.0', 'B3D_5.0' };
%%
stackSize = [616, 432, 100];
numFiles = size(compressionLevels, 2);
%% Read files
Data = zeros([stackSize, numFiles]);
%%
for i = 1:numFiles
    data = h5read([baseFolder, sprintf(fileNameFormat, compressionLevels(i))],'/exported_data');
    data = data(2,:,:,:) > 0.5;
    Data(:,:,:,i) = reshape(data, stackSize);
end

%% Calculate Sorensen-Dice similarity coefficients
QS = zeros(numFiles,1);

for i = 1:numFiles
        QS(i,1) = sorensenDice(Data(:,:,:,i), Data(:,:,:,1));
end

%% make overlap figure
%% q = 1
%makeRectangle(282, 186, 64, 64);
orig = Data(282:282+64,186:186+64,40,1);
compr1 = Data(282:282+64,186:186+64,40,4);
compr4 = Data(282:282+64,186:186+64,40,9);
overlap1 = (orig + 2*compr1)';
overlap4 = (orig + 2*compr4)';
imagesc(overlap4)
axis equal
%%
cmap = [0, 0, 0; 0 1 0; 1 0 1; 1 1 1];
imwrite(uint8(overlap1), cmap, [saveFolder, 'Fig2b.png']);
imwrite(uint8(overlap4), cmap, [saveFolder, 'Fig2c.png']);

%% get compression ratios
compressionRatios = zeros(size(compressionLevels));
compressedFileName = 'drosophila_1_masked24_1024x2048x211_filtered.h5';
datasetFormat = 'B3D_%.2f';
%%
fid = H5F.open([baseFolder compressedFileName]);
for i = 1:numFiles    
    dset_id = H5D.open(fid, sprintf(datasetFormat, compressionLevels(i)));
    compressedSize = H5D.get_storage_size(dset_id);
    space_id = H5D.get_space(dset_id);
    [ndims,h5_dims] = H5S.get_simple_extent_dims(space_id);
    fullSize = prod(h5_dims);
    H5S.close(space_id);
    type_id = H5D.get_type(dset_id);
    type_size = H5T.get_size(type_id);
    H5T.close(type_id);
    H5D.close(dset_id);    
    compressionRatios(i) = type_size*fullSize/compressedSize;
end
H5F.close(fid);
%%

%% plot results
figureSize = 300;
borderWidth = 1.5;
lineWidth = 1.5;
msMult = 6;
scheme = 'RdYlBu';
colors = flip(brewermap(5, scheme));
%%
f33 = figure(33);
cla reset;
pos = get(f33, 'Position');
set(f33, 'Position', [pos(1), pos(2), figureSize*1.4, figureSize])
hold on
xx = 0:.1:5;
interp = spline(compressionLevels, QS, xx);
plot(xx,interp, 'LineWidth', lineWidth, 'Color', colors(end,:));
plot(compressionLevels, QS, '+', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(end,:))
axis square
set(gcf, 'Color', 'w')
set(gca, 'LineWidth', 1)
h = gca;
h.YRuler.LineWidth = borderWidth;
h.XRuler.LineWidth = borderWidth;
h.GridColor = g(1);
grid on
set(gca, 'Box', 'on', 'Color', g(0.9), 'FontSize', 14)


xlabel('compression level')
ylabel('overlap score', 'Color', colors(end,:))
% set(gca, 'XTick', 0:5, 'YTick', 0.92:0.02:1,'YTickLabels', {'0.92', '', '0.96', '', '1'})
set(gca, 'XTick', 0:5, 'YTick', 0.9:0.025:1,'YTickLabels', {'0.9', '', '0.95', '', '1'})
ylim([0.89,1.01])
xlim([0,5])

ax1_pos = h.Position; % position of first axes
ax2 = axes('Position',ax1_pos,...
    'XAxisLocation','bottom',...
    'YAxisLocation','right',...
    'Color','none');
set(ax2, 'Box', 'on', 'FontSize', 14)
set(ax2, 'XTick', 0:5, 'YTick', 0:60:120)
hold on
interp2 = spline(compressionLevels, compressionRatios, xx);
plot(xx,interp2, 'LineWidth', lineWidth, 'Color', colors(1,:));
plot(compressionLevels, compressionRatios, '+', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(1,:))

xlim([0,5])
ylim([-12, 132])
ylabel('compression ratio', 'Color', colors(1,:))
axis square
%% save plot
% requires export_fig function 
% can be downloaded from: https://de.mathworks.com/matlabcentral/fileexchange/23629-export-fig
export_fig([saveFolder, '\Fig2d.png'])
export_fig([saveFolder, '\Fig2d.pdf'])


