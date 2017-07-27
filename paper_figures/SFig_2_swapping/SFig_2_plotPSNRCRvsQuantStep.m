%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Supplementary Figure 2a,b for B³D paper                        %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 27.06.2017                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear variables;
%%
baseFolder = 'D:\GPU_compression\!paper_figures\SFig_2_swapping\';

%% read compressed files
files = {'drosophila_1_masked24_1024x2048x211_filtered_101.h5', 'drosophila_1_masked24_1024x2048x211_filtered_102.h5'};
compressionLevels = [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5];
Nc = length(compressionLevels);
Nf = length(files);
datasetFormat = '/B3D_%.2f';

%%
P = zeros(Nf, Nc);
CR = zeros(Nf, Nc);
%%
H5.get_libversion;  % this is only here to initialize the H5 library an enable dynamically loaded filters
orig = h5read([baseFolder, files{1}], sprintf(datasetFormat, compressionLevels(1)));
%%
for f = 1:Nf
    for c = 1:Nc
        %% get compression ratio
        fid = H5F.open([baseFolder, files{f}]);
        dset_id = H5D.open(fid, sprintf(datasetFormat, compressionLevels(c)));
        compressedSize = H5D.get_storage_size(dset_id);
        space_id = H5D.get_space(dset_id);
        [ndims,h5_dims] = H5S.get_simple_extent_dims(space_id);
        fullSize = prod(h5_dims);
        H5S.close(space_id);
        type_id = H5D.get_type(dset_id);
        type_size = H5T.get_size(type_id);
        H5T.close(type_id);
        H5D.close(dset_id);    
        CR(f,c) = type_size*fullSize/compressedSize;
        H5F.close(fid);
        
        %% read dataset
        data = h5read([baseFolder, files{f}], sprintf(datasetFormat, compressionLevels(c)));
        
        %% calculate PSNR
        P(f,c) = psnr(orig, data, 65535);
    end
end


%% plot stuff
figureSize = 300;
borderWidth = 1.5;
lineWidth = 1.5;
fontSize = 14;
msMult = 5;
scheme = 'RdYlBu';
colors = flip(brewermap(5, scheme));
%% PSNR
f41 = figure(41);
cla reset;
pos = get(f41, 'Position');
set(f41, 'Position', [pos(1), pos(2), figureSize*1.4, figureSize])
hold on
xx = 0.05:.1:5;

interp = spline(compressionLevels(2:end), P(1,2:end), xx);
plot(xx,interp, 'LineWidth', lineWidth, 'Color', colors(end,:));
interp = spline(compressionLevels(2:end), P(2,2:end), xx);
plot(xx,interp, 'LineWidth', lineWidth, 'Color', colors(1,:));

plot(compressionLevels, P(1,:), '+', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(end,:))
plot(compressionLevels, P(2,:), '+', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(1,:))

axis square
set(gcf, 'Color', 'w')
set(gca, 'LineWidth', 1)
h = gca;
h.YRuler.LineWidth = borderWidth;
h.XRuler.LineWidth = borderWidth;
h.GridColor = g(1);
grid on
set(gca, 'Box', 'on', 'Color', g(0.9), 'FontSize', 14)

leg = legend({'Mode 1', 'Mode 2'}, 'Location', 'NorthEast');
set(leg, 'Color', 'w')
xlabel('compression level')
ylabel('PSNR')
% title('Drosophila nucleus segmentation')
set(gca, 'XTick', 0:5)
% ylim([0.89,1.01])
xlim([0,5])
%%
export_fig([baseFolder, 'SFig_2_swapping_PSNR.pdf'])
%% compression ratio
f42 = figure(42);
cla reset;
pos = get(f42, 'Position');
set(f42, 'Position', [pos(1), pos(2), figureSize*1.4, figureSize])
hold on
xx = 0:.1:5;

interp = spline(compressionLevels, CR(1,:), xx);
plot(xx,interp, 'LineWidth', lineWidth, 'Color', colors(end,:));
interp = spline(compressionLevels, CR(2,:), xx);
plot(xx,interp, 'LineWidth', lineWidth, 'Color', colors(1,:));

plot(compressionLevels, CR(1,:), '+', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(end,:))
plot(compressionLevels, CR(2,:), '+', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(1,:))

axis square
set(gcf, 'Color', 'w')
set(gca, 'LineWidth', 1)
h = gca;
h.YRuler.LineWidth = borderWidth;
h.XRuler.LineWidth = borderWidth;
h.GridColor = g(1);
grid on
set(gca, 'Box', 'on', 'Color', g(0.9), 'FontSize', 14)

leg = legend({'Mode 1', 'Mode 2'}, 'Location', 'NorthWest');
set(leg, 'Color', 'w')
xlabel('compression level')
ylabel('compression ratio')
% title('Drosophila nucleus segmentation')
set(gca, 'XTick', 0:5)
% ylim([0.89,1.01])
xlim([0,5])

%%
export_fig([baseFolder, '\SFig_2_swapping_CR.pdf'])