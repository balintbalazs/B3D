@echo off
SETLOCAL EnableDelayedExpansion 

set filename=drosophila_1_masked24_1024x2048x211_filtered

set datasets[0]=B3D_0.00_even_smaller
set datasets[1]=B3D_0.25_even_smaller
set datasets[2]=B3D_0.50_even_smaller
set datasets[3]=B3D_1.00_even_smaller
set datasets[4]=B3D_1.50_even_smaller
set datasets[5]=B3D_2.00_even_smaller
set datasets[6]=B3D_2.50_even_smaller
set datasets[7]=B3D_3.00_even_smaller
set datasets[8]=B3D_4.00_even_smaller
set datasets[9]=B3D_5.00_even_smaller

for /F "tokens=2 delims==" %%s in ('set datasets[') DO (
	echo %%s
	"C:\Program Files\ilastik-1.2.0rc10\ilastik.exe" --headless --project=nucleus_segmenting.ilp --output_filename_format={dataset_dir}/%filename%_%%s_Probabilities.h5 %filename%.h5/%%s
)