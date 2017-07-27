/*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate subfigures for Supplementary Figure 3 for B³D paper            %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 27.06.2017                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/
run("Scriptable load HDF5...", "load=D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\drosophila_1_masked24_1024x2048x211_filtered.h5 datasetnames=/B3D_0.00 nframes=1 nchannels=1");
rename("lossless");
run("Duplicate...", "title=lossless-194 duplicate range=194-194");
selectWindow("lossless");
close();
selectWindow("lossless-194");
makeRectangle(318, 528, 384, 864); // zoom 0
run("Crop");
setMinAndMax(200, 1316);
run("RGB Color");
run("Save", "save=D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\lossless-194.tif");
makeRectangle(183, 325, 64, 64); // zoom 1
run("Duplicate...", "title=lossless-194-zoom1.tif");
run("Save", "save=D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\lossless-194-zoom1.tif");
makeRectangle(21, 5, 24, 24); // zoom 2
run("Duplicate...", "title=lossless-194-zoom2.tif");
selectWindow("lossless-194-zoom2.tif");
run("Save", "save=D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\lossless-194-zoom2.tif");
saveAs("PNG", "D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\lossless-194-zoom2.png");
close();

selectWindow("lossless-194-zoom1.tif");
setForegroundColor(255, 255, 0);
run("Draw", "slice");
saveAs("PNG", "D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\lossless-194-zoom1.png");
close();

selectWindow("lossless-194.tif");
setForegroundColor(255, 255, 0);
makeRectangle(183, 325, 64, 64);
run("Draw", "slice");
makeRectangle(182, 324, 66, 66);
run("Draw", "slice");
makeRectangle(181, 323, 68, 68);
run("Draw", "slice");
makeRectangle(180, 322, 70, 70);
run("Draw", "slice");
saveAs("PNG", "D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\lossless-194.png");
close();

run("Scriptable load HDF5...", "load=D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\drosophila_1_masked24_1024x2048x211_filtered.h5 datasetnames=/B3D_1.00 nframes=1 nchannels=1");
rename("wnl");
run("Duplicate...", "title=wnl-194 duplicate range=194-194");
selectWindow("wnl");
close();
selectWindow("wnl-194");
makeRectangle(318, 528, 384, 864); // zoom 0
run("Crop");
setMinAndMax(200, 1316);
run("RGB Color");
run("Save", "save=D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\wnl-194.tif");
makeRectangle(183, 325, 64, 64); // zoom 1
run("Duplicate...", "title=wnl-194-zoom1.tif");
run("Save", "save=D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\wnl-194-zoom1.tif");
makeRectangle(21, 5, 24, 24); // zoom 2
run("Duplicate...", "title=wnl-194-zoom2.tif");
selectWindow("wnl-194-zoom2.tif");
run("Save", "save=D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\wnl-194-zoom2.tif");
saveAs("PNG", "D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\wnl-194-zoom2.png");
close();

selectWindow("wnl-194-zoom1.tif");
setForegroundColor(255, 255, 0);
run("Draw", "slice");
saveAs("PNG", "D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\wnl-194-zoom1.png");
close();

selectWindow("wnl-194.tif");
setForegroundColor(255, 255, 0);
run("Draw", "slice");
makeRectangle(182, 324, 66, 66);
run("Draw", "slice");
makeRectangle(181, 323, 68, 68);
run("Draw", "slice");
makeRectangle(180, 322, 70, 70);
run("Draw", "slice");
saveAs("PNG", "D:\\GPU_compression\\!paper_figures\\SFig_3_imageQuality\\wnl-194.png");
close();