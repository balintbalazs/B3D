%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate blinking fluorophores on a regular grid                        %
% Author: Balint Balazs                                                   %
% contact: balint.balazs@embl.de                                          %
% 15.05.2017                                                              %
% EMBL Heidelberg, Cell Biology and Biophysics                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameters
spacing = 21.1; % fluorophore spacing
gridSize = 10; % grid size
Cam_Bg = 10; % camera background offset in DN
pixsize = 1; % um
sig = 1; % sigma in number of pix
gauss = 0; % gaussian noise 
S = round(spacing*11); % image size
Nim = 10000; % number of images generated

NPH = [100,500,1000,5000,10000,50000,5000,50000]; % mean number of photons per molecule
BG = [20,20,20,20,20,20,100,200]; % background illumination in photons
N = size(NPH, 2);


%%
for I=2:6
%     for J=1:N
tic
        Nph = NPH(I);
        Nph_bg = BG(I);
        %% generating coordinate system
        x = 1:S;
        y = x;

        [X, Y] = meshgrid(x,y);

        %%
        Z = zeros(S,S);
        for i=1:gridSize
            for j=1:gridSize
                Z = Z + ...
                    0.5 .* ( erf((X-spacing*i+0.5)/sqrt(2*sig^2)) - erf((X-spacing*i-0.5)/sqrt(2*sig^2)) ) .* ...
                    0.5 .* ( erf((Y-spacing*j+0.5)/sqrt(2*sig^2)) - erf((Y-spacing*j-0.5)/sqrt(2*sig^2)) );
            end
        end


        %% scaling by number of photons
        GndTruth = Z*Nph;

        %% adding illumination background
        GndTruth = GndTruth + Nph_bg;

        %% Shot noise (poisson)
        im = zeros(S,S,Nim);
        for i=1:Nim
            im(:,:,i) = poissrnd(GndTruth);
        end

        %% image background
        im = im + Cam_Bg;

        %% Camera noise (gaussian)
        im = im + gauss*randn(S,S,Nim);

        %% save generated stack
        fn = ['spotsSimulation_NPH', num2str(Nph),'_BG', num2str(Nph_bg), '.h5'];
        h5create(fn, '/Data', size(im), 'Datatype', 'uint16')
        h5write(fn, '/Data', im);
%     end
toc
end
    
% end
% %%
% figure(1)
% imagesc(GndTruth);
% colorbar
% axis image






