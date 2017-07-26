@echo off

set folder=%~dp0
IF %folder:~-1%==\ SET folder=%folder:~0,-1%
rem echo %folder%

if defined HDF5_PLUGIN_PATH (
	echo HDF5_PLUGIN_PATH defined
	if "%folder%"=="%HDF5_PLUGIN_PATH%" (
		echo Everything is already set up.
	) else (
		echo Copying dlls to %HDF5_PLUGIN_PATH%
		copy /-Y "%folder%\*.dll" "%HDF5_PLUGIN_PATH%"
	)
) else (
	echo HDF5_PLUGIN_PATH not defined. Setting it to current directory.
	setx HDF5_PLUGIN_PATH "%folder%"
)
pause