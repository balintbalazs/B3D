@echo off

type LICENSE
echo(
:ask
echo Please take a moment to read the above license agreement now. Do you
echo agree, and wish to proceed with the installation? (Y/N)
set INPUT=
set /P INPUT=%=%
If /I "%INPUT%"=="y" goto yes 
If /I "%INPUT%"=="n" goto no
goto ask
:yes
set folder=%~dp0
IF %folder:~-1%==\ SET folder=%folder:~0,-1%
rem echo %folder%

if defined HDF5_PLUGIN_PATH (
	echo HDF5_PLUGIN_PATH defined
	if "%folder%"=="%HDF5_PLUGIN_PATH%" (
		echo Everything is already set up.
	) else (
		if exist "%HDF5_PLUGIN_PATH%" (
			echo Copying dlls to %HDF5_PLUGIN_PATH%
			copy /-Y "%folder%\*.dll" "%HDF5_PLUGIN_PATH%"
		) else (
			echo HDF5_PLUGIN_PATH is defined, but location is not found. Setting it to current directory.
			setx HDF5_PLUGIN_PATH "%folder%"
		)
	)
) else (
	echo HDF5_PLUGIN_PATH not defined. Setting it to current directory.
	setx HDF5_PLUGIN_PATH "%folder%"
)
echo B3D is successfully installed.
goto end
:no
echo Quitting installation.
:end
pause