/*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare stacks for segmentation                                         %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 23.06.2017                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/
baseFolder = "D:\\GPU_compression\\!paper_figures\\Fig_2_drosophila\\"
files = newArray("drosophila_1_masked24_1024x2048x211_filtered.h5")
//datasets = newArray("B3D_1.0", "B3D_3.0","B3D_swapped_1.0", "B3D_swapped_3.0")
datasets = newArray("B3D_0.00", "B3D_0.25", "B3D_0.50", "B3D_1.00", "B3D_1.50", "B3D_2.00", "B3D_2.50", "B3D_3.00", "B3D_4.00", "B3D_5.00")

for (i=0; i<files.length; i++) {
	for (j=0; j<datasets.length; j++) {
		run("Scriptable load HDF5...", "load=" + baseFolder + files[i] + " datasetnames=/"+datasets[j]+" nframes=1 nchannels=1");
		run("Slice Remover", "first=41 last=211 increment=1");
		run("Slice Remover", "first=1 last=5 increment=1");
		makeRectangle(198, 82, 616, 1834);
		run("Crop");
		run("Properties...", "channels=1 slices=35 frames=1 unit=micrometer pixel_width=0.26 pixel_height=0.26 voxel_depth=1");	
		run("Scale...", "x=1.0 y=1.0 z=3.8461538461538461538461538461538 width=616 height=1834 depth=134 interpolation=Bilinear average process create title=[Data_small-1]");
		run("Properties...", "channels=1 slices=134 frames=1 unit=micrometer pixel_width=0.26 pixel_height=0.26 voxel_depth=0.2611940");
		//selectWindow("uncompressed_small-1");
		//run("Scriptable save HDF5 (append)...", "save=" + baseFolder + files[i]+" dsetnametemplate=/"+datasets(j)+"_small formattime=%d formatchannel=%d compressionlevel=0");
		
		run("Slice Remover", "first=101 last=134 increment=1");
		makeRectangle(0, 822, 616, 432);
		run("Crop");
		run("Scriptable save HDF5 (append)...", "save="+ baseFolder + files[i] + " dsetnametemplate=/"+datasets[j]+"_even_smaller formattime=%d formatchannel=%d compressionlevel=0");
		
		close();
		//selectWindow("C:\Data\GPU_compression\noise_vs_compression\drosophila_1_1024x2048x211_decompressed.h5: uncompressed_small");
		close();	
	}
}