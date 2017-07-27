%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Figure 2 and Supplementary Figure 6 for B³D paper              %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 23.06.2017                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear variables;
%%
baseFolder = 'D:\GPU_compression\STORM_From_Joran\simulateLocalizationData\newSimulations10000\compressed_102\';

Nphotons = [500,1000,5000,10000,50000];
BGlevel = 20;
compressionLevels = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4];

pixelSizeNm = 100;
%%
fileFormat = 'spotsSimulation_NPH%d_BG%d_B3D%.2f_sml.mat';
compressedFileFormat = 'spotsSimulation_NPH%d_BG%d_B3D%.2f.h5';

%%
cmax = size(compressionLevels,2);
Nmax = size(Nphotons,2);
%% get compression ratio
compressionRatios = zeros(cmax, Nmax);
for c = 1:cmax
    for N = 1:Nmax
        fid = H5F.open([baseFolder sprintf(compressedFileFormat, Nphotons(N), BGlevel, compressionLevels(c))]);
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
        compressionRatios(c,N) = type_size*fullSize/compressedSize;
        H5F.close(fid);
    end
end
%%
locprec = zeros(cmax, Nmax);
locprecFromFitting = locprec;
%%
for c =1:cmax
    for N = 1:Nmax
        %%
        load([baseFolder, sprintf(fileFormat, Nphotons(N), BGlevel, compressionLevels(c))]);

        locX = saveloc.loc.xnm;
        locY = saveloc.loc.ynm;

        errX = mod(locX, 2110);
        errX(errX > 1055) = errX(errX > 1055) - 2110;
        errX = errX(abs(errX)<300);
        errY = mod(locY, 2110);
        errY(errY > 1055) = errY(errY > 1055) - 2110;
        errY = errY(abs(errY)<300);
        
%         errD = sqrt(errX .* errX + errY.* errY);
        %%
        locprec(c,N) = std(errX);
        locprecFromFitting(c,N) = mean(saveloc.loc.xerr);
    end
end
%%
for N=1:Nmax
    temp = cramer_rao(BGlevel, Nphotons(N));
    cr(N) = temp(1)*pixelSizeNm;
end

%%
crNormLocPrec = bsxfun(@rdivide, locprec, cr);
normLocPrec = bsxfun(@rdivide, locprec, locprec(1,:));
normLocPrecFromFitting = bsxfun(@rdivide, locprec, locprecFromFitting(1,:));

%% plot stuff
figureSize = 300;
borderWidth = 1.5;
lineWidth = 1.5;
msMult = 6;
scheme = 'RdYlBu';
colors = flip(brewermap(5, scheme));
xmax = 4;
column = 3;
%%
f31 = figure(1175);
cla reset;
pos = get(f31, 'Position');
set(f31, 'Position', [pos(1), pos(2), figureSize*1.4, figureSize])
hold on
xx = 0:.1:xmax;
interp = spline(compressionLevels, crNormLocPrec(:,column)', xx);
plot(xx,interp, 'LineWidth', lineWidth, 'Color', colors(end,:));
plot(compressionLevels, crNormLocPrec(:,column)', '+', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(end,:))
% plot(xx, sqrt(1+xx.^2/12), 'g-', 'LineWidth', 2*lineWidth)
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
ylabel('relative localization error', 'Color', colors(end,:));
set(gca, 'XTick', 0:xmax, 'YTick', 1:0.1:1.6,'YTickLabels', {'1', '', '1.2', '', '1.4', '', '1.6'})
ylim([0.94,1.66])
xlim([0,xmax])

ax1_pos = h.Position; % position of first axes
ax2 = axes('Position',ax1_pos,...
    'XAxisLocation','bottom',...
    'YAxisLocation','right',...
    'Color','none');
set(ax2, 'Box', 'on', 'FontSize', 14)
set(ax2, 'XTick', [], 'YTick', 0:5:15)
hold on
interp2 = spline(compressionLevels, compressionRatios(:,column), xx);
plot(xx,interp2, 'LineWidth', lineWidth, 'Color', colors(1,:));
plot(compressionLevels, compressionRatios(:,column), '+', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(1,:))

xlim([0,xmax])
ylim([-1.5, 16.5])
ylabel('compression ratio', 'Color', colors(1,:))
axis square
%%
saveFolder = fileparts(matlab.desktop.editor.getActiveFilename);
export_fig([saveFolder, '\locprecVSquantStep_Fig2h_Joran.png'])
export_fig([saveFolder, '\locprecVSquantStep_Fig2h_Joran.pdf'])



%% plot multiple
figureSize = 500;
colors = brewermap(9, 'set1');
%%
f1171 = figure(1171);
cla reset;
pos = get(f1171, 'Position');
set(f1171, 'Position', [pos(1), pos(2), figureSize*1.4, figureSize])
hold on
xx = 0:.1:xmax;
interp = spline(compressionLevels, crNormLocPrec', xx);
for N=2:Nmax
    %plot(xx,interp(N,:), 'LineWidth', lineWidth, 'Color', colors(N,:));
    plot(compressionLevels, crNormLocPrec(:,N)', '-', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(N-1,:))
end
axis square
set(gcf, 'Color', 'w')
set(gca, 'LineWidth', 1)
h = gca;
h.YRuler.LineWidth = borderWidth;
h.XRuler.LineWidth = borderWidth;
h.GridColor = g(1);
grid on
set(gca, 'Box', 'on', 'Color', g(0.9), 'FontSize', 14)
leg1171 = legend({'1000,','5000','10000','50000'}, 'Location', 'NorthWest');
set(leg1171, 'Color', 'w');
xlabel('compression level')
ylabel('relative localization error');
set(gca, 'XTick', 0:xmax, 'YTick', 1:0.2:1.8,'YTickLabels', {'1', '', '1.4', '', '1.8'})
ylim([0.92,1.88])
xlim([0,xmax])

%% plot multiple
figureSize = 500;
f1172 = figure(1172);
cla reset;
pos = get(f1172, 'Position');
set(f1172, 'Position', [pos(1), pos(2), figureSize*1.4, figureSize])
hold on
for c=1:cmax-1
    %%plot(xx,interp(N,:), 'LineWidth', lineWidth, 'Color', colors(N,:));
    plot(Nphotons, crNormLocPrec(c,:), '-', 'LineWidth', 2*lineWidth, 'MarkerSize', msMult*lineWidth, 'Color', colors(c,:))
end
axis square
set(gcf, 'Color', 'w')
set(gca, 'LineWidth', 1)
h = gca;
h.YRuler.LineWidth = borderWidth;
h.XRuler.LineWidth = borderWidth;
h.GridColor = g(1);
grid on
set(gca, 'Box', 'on', 'Color', g(0.9), 'FontSize', 14)
set(gca, 'Xscale', 'log')

xlabel('number of photons/localizaiton')
ylabel('relative localization error');
set(gca, 'XTick', Nphotons, 'YTick', 1:0.1:1.4)
ylim([0.96,1.44])
xlim([500,50000])

leg = legend({'lossless','0.25\sigma','0.5\sigma','0.75\sigma', '1\sigma','1.5\sigma', '2\sigma', '2.5\sigma','3\sigma'}, 'Location', 'NorthEastOutside');
set(leg, 'Color', 'w', 'LineWidth', borderWidth);
%%
saveFolder = fileparts(matlab.desktop.editor.getActiveFilename);
export_fig([saveFolder, '\locprecVsNphotons_SFig_6.png'])
export_fig([saveFolder, '\locprecVsNphotons_SFig_6.pdf'])