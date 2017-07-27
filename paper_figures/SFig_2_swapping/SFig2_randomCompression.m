%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Supplementary Figure 2c,d for B³D paper                           %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 27.06.2017                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear variables;
%%
baseFolder = 'D:\GPU_compression\!paper_figures\SFig_2_swapping\';
%%
m = 10;
s = [64,64,10000];
%% generate random image
poiss = ones(s) * m;
poiss = poissrnd(poiss);
poiss = uint16(poiss);
fn = [baseFolder, 'poiss_10']
writeRaw3D(fn, poiss);
%% do compression

%% read compressed files
files = {'poiss_10_64x64x10000_filtered_101.h5', 'poiss_10_64x64x10000_filtered_102.h5'};
compressionLevels = [0, 1, 2, 3, 4, 5];
Nc = length(compressionLevels);
Nf = length(files);
datasetFormat = '/B3D_%.2f';

%%
Poiss = zeros([s, Nf, Nc]);
PSNR = zeros(Nf, Nc);
CR = zeros(Nf, Nc);
%%
Corr = zeros([2*s(1)-1, 2*s(2)-1, Nf, Nc]);
H5.get_libversion;  % this is only here to initialize the H5 library an enable dynamically loaded filters
%%
depth = s(3)
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
        Poiss(:,:,:,f,c) = h5read([baseFolder, files{f}], sprintf(datasetFormat, compressionLevels(c)));
        
        %% calculate PSNR
        PSNR(f,c) = psnr(Poiss(:,:,:,f,1), Poiss(:,:,:,f,c), 256);
        
        %% subtract mean
        Poiss(:,:,:,f,c) = Poiss(:,:,:,f,c) - mean(mean(mean(Poiss(:,:,:,f,c))));
%         %% calcualte correlation
%         tic
%         for j = 1:s(3)
%             Corr(:,:,f,c) = Corr(:,:,f,c) + xcorr2(Poiss(:,:,j,f,c),Poiss(:,:,j,f,c));
%         end
%         toc
    end
end
Corr = Corr/s(3);


%% plot stuff
figure(2)
cmap = brewermap(256,'RdYlBu');
titleFormat = 'Mode %d, q=%.2f';
for f =1:Nf
    for c = 1:Nc
        subplot(Nf,Nc,(f-1)*Nc+c)
        imagesc(Corr(:,:,f,c),[-100,7e3]);
        axis image
        colorbar
        title([sprintf(titleFormat, f, compressionLevels(c)), '\sigma']);
        colormap(cmap)
    end
end

%% make figures
%%
figureSize = 700;
borderWidth = 1.5;
lineWidth = 1.5;
scheme = 'RdYlBu';
colors = brewermap(256,scheme);
dispRange = [-100, 7e3];
dispSize = 17;
fullSize = size(Corr, 1);
%% bar plot ratios
f = figure(1);
cla reset;
pos = get(f, 'Position');
set(f, 'Position', [pos(1), pos(2), figureSize, figureSize])
set(gcf, 'Color', 'w')
set(gca, 'LineWidth', 1)
h = gca;
h.YRuler.LineWidth = borderWidth;
h.XRuler.LineWidth = borderWidth;
colormap(colors)
%%
subplot(2,2,1)
imagesc(Corr((fullSize-dispSize)/2+1:(fullSize+dispSize)/2,(fullSize-dispSize)/2+1:(fullSize+dispSize)/2,1,2), dispRange)
colorbar
axis image
title('Mode 1, q=1\sigma')

subplot(2,2,2)
imagesc(Corr((fullSize-dispSize)/2+1:(fullSize+dispSize)/2,(fullSize-dispSize)/2+1:(fullSize+dispSize)/2,1,5), dispRange)
colorbar
axis image
title('Mode 1, q=4\sigma')

subplot(2,2,3)
imagesc(Corr((fullSize-dispSize)/2+1:(fullSize+dispSize)/2,(fullSize-dispSize)/2+1:(fullSize+dispSize)/2,2,2), dispRange)
colorbar
axis image
title('Mode 2, q=1\sigma')

subplot(2,2,4)
imagesc(Corr((fullSize-dispSize)/2+1:(fullSize+dispSize)/2,(fullSize-dispSize)/2+1:(fullSize+dispSize)/2,2,5), dispRange)
colorbar
axis image
title('Mode 2, q=4\sigma')

%% save pdf
export_fig([baseFolder, 'SFig_2_correlations.pdf'])

