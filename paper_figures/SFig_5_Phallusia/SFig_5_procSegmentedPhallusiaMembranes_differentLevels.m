%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare phallusia membrane segmentation                                 %
% on original and B³D compressed images                                   %
% Cheking overlap of membranes                                            %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 24.10.2016                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% draw membrane around cells
clear variables;
%%
baseFolder = 'D:\GPU_compression\!paper_figures\SFig_5_Phallusia\';
origFile = 'seg_MARS\seg_t005';

compressionLevels = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5];
numFiles = size(compressionLevels, 2);

fileNameFormat = 'decompressed\\%d\\seg_MARS\\seg_t005.inr';
outFileNameFormat = 'membranes\\seg_t005_%d_2px.h5';
%%
for i = 1:numFiles
    %% load file
    data = readINR([baseFolder, sprintf(fileNameFormat, compressionLevels(i)*1000)]);
    %% define structuring element
    % radius = 1
    % 6 connected - only faces
    %%nbhd = cat(3,[0,0,0;0,1,0;0,0,0],[0,1,0;1,1,1;0,1,0],[0,0,0;0,1,0;0,0,0]);
    %%se = strel(nbhd);

    %% radius = 2
    a = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0];
    b = [0,0,0,0,0; 0,0,1,0,0; 0,1,1,1,0; 0,0,1,0,0; 0,0,0,0,0];
    c = [0,0,1,0,0; 0,1,1,1,0; 1,1,1,1,1; 0,1,1,1,0; 0,0,1,0,0];
    se = cat(3,a,b,c,b,a);
    %% define background value and determine number of cells
    bgValue = 1;
    numCells = max(data(:)) - bgValue;

    %% create result and temporary arrays
    proc = data;
    temp = zeros(size(data));

    %% erode each cell and use difference as membrane
    tic
    for c = bgValue+1:bgValue+numCells
       temp = data == c;
       temp = imerode(temp,se);
       temp = xor(temp,data==c);
       proc = proc .* uint16(~temp);
    end
    toc
    % 244.32 s for r=1
    % 578.01 s for r=2
    %% save results
    fileNameH5 = [baseFolder, sprintf(outFileNameFormat, compressionLevels(i)*1000)];
    try
        h5create(fileNameH5, '/Data+membrane', size(proc),...
            'Datatype','uint8','ChunkSize',size(proc),'Deflate',6);
    catch err
        if (~strcmp(err.identifier,'MATLAB:imagesci:h5create:datasetAlreadyExists'))
            rethrow(err)
        end
    end
    h5write(fileNameH5, '/Data+membrane', proc);
    %% remove outer membrane by dilating background and masking membranes
    bg = data == 1;
    bg = imdilate(bg,se);
    proc = proc .* uint16(~bg);
    proc = proc + uint16(bg);
    
%% save again without outer membrane for visualization
    % dataset name should be "/Data" because of Fiji macro
    try
        h5create(fileNameH5, '/Data', size(proc),...
            'Datatype','uint8','ChunkSize',size(proc),'Deflate',6);
    catch err
        if (~strcmp(err.identifier,'MATLAB:imagesci:h5create:datasetAlreadyExists'))
            rethrow(err)
        end
    end
    h5write(fileNameH5, '/Data', proc);
end

%%

%%
% baseFolder = 'D:\GPU_compression\Phallusia_from_Ulla\PHcitrinemovie1_4h30\2016-04-08_14.29.36_original\';
origFile = 'differentLevelsMasked\membranes\seg_t005_0_2px.h5';
origM = h5read([baseFolder, sprintf(outFileNameFormat, 0)], '/Data+membrane');
origM = (origM == 0);
 %%
QS = zeros(1,numFiles);
for i = 1:numFiles   
    %%
    b3dM = h5read([baseFolder, sprintf(outFileNameFormat, compressionLevels(i)*1000)], '/Data+membrane');
    b3dM = (b3dM == 0);   
    [origM, b3dM] = matchCropRegion(origM, b3dM);
    QS(i) = 2 * sum(origM(:) & b3dM(:)) / (sum(origM(:)) + sum(b3dM(:)));
end

%% get compression ratios
stacks = {'S0', 'S1'};
cams = {'CL', 'CR'};
compressedFileFormat = 'D:\\GPU_compression\\!paper_figures\\SFig_5_Phallusia\\compressed\\%d\\%s%s\\phallusia_T05_%s%s_00005.h5';
CR = zeros(1,length(compressionLevels));
%%
for q = 1:length(compressionLevels)
    for s = 1:length(stacks)
        for c = 1:length(cams)
            fid = H5F.open(sprintf(compressedFileFormat, compressionLevels(q)*1000, stacks{s}, cams{c}, stacks{s}, cams{c}));
            dset_id = H5D.open(fid, '/Data');
            compressedSize = H5D.get_storage_size(dset_id);
            space_id = H5D.get_space(dset_id);
            [ndims,h5_dims] = H5S.get_simple_extent_dims(space_id);
            fullSize = prod(h5_dims);
            H5S.close(space_id);
            type_id = H5D.get_type(dset_id);
            type_size = H5T.get_size(type_id);
            H5T.close(type_id);
            H5D.close(dset_id);    
            CR(q) = CR(q) + type_size*fullSize/compressedSize;
            H5F.close(fid);
        end
    end
end
CR = CR ./ 4;

%% plot stuff
figureSize = 300;
borderWidth = 1.5;
lineWidth = 1.5;
msMult = 4;
scheme = 'RdYlBu';
colors = flip(brewermap(5, scheme));
%%
f32 = figure(32);
cla reset;
pos = get(f32, 'Position');
set(f32, 'Position', [pos(1), pos(2), figureSize*1.4, figureSize])
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
ylabel('overlap score')
% set(gca, 'XTick', 0:5, 'YTick', 0.92:0.02:1,'YTickLabels', {'0.92', '', '0.96', '', '1'})
set(gca, 'XTick', 0:5, 'YTick', 0.8:0.05:1,'YTickLabels', {'0.8', '', '0.9', '', '1'})
ylim([0.78,1.02])
xlim([0,5])

ax1_pos = h.Position; % position of first axes
ax2 = axes('Position',ax1_pos,...
    'XAxisLocation','bottom',...
    'YAxisLocation','right',...
    'Color','none');
set(ax2, 'Box', 'on', 'FontSize', 14)
set(ax2, 'XTick', 0:5, 'YTick', 0:100:200)
hold on
interp2 = spline(compressionLevels, CR, xx);
plot(xx,interp2, 'LineWidth', lineWidth, 'Color', colors(1,:));
plot(compressionLevels, CR, '+', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(1,:))

xlim([0,5])
ylim([-20, 220])
ylabel('compression ratio')
axis square
%%
saveFolder = fileparts(matlab.desktop.editor.getActiveFilename);
export_fig([saveFolder, '\phallusiaOverlapVSquantStep.png'])
export_fig([saveFolder, '\phallusiaOverlapVSquantStep.pdf'])