function CR = cramer_rao(thetaBg, thetaIo)
%% calculate Cramer-Rao lower bound for ground truth
%  based on Fast, single-molecule localization that achieves
%  theoretically minimum uncertainty 
%  Smith et al. Nature Methods 7, 373 - 375 (2010)
%  using radius of 3*sigma, full box size = 2*3*sigma+1
% size(ThetaBg, 2);
% CR = zeros(size(ThetaBg));
% for k=1:size(ThetaBg, 2)
sigma = 1;
radius = 4*sigma;
x = -radius:radius;
y = x;

[X, Y] = meshgrid(x, y);

thetaX = 0;
thetaY = 0;
% thetaIo = 1000;
% thetaBg = ThetaBg(k);

deltaEx = 0.5 .* ( erf((X-thetaX+0.5)/sqrt(2*sigma^2)) - erf((X-thetaX-0.5)/sqrt(2*sigma^2)) );
deltaEy = 0.5 .* ( erf((Y-thetaY+0.5)/sqrt(2*sigma^2)) - erf((Y-thetaY-0.5)/sqrt(2*sigma^2)) );
MU = thetaIo .* deltaEx .* deltaEy + thetaBg;

%imagesc(MU); colorbar;

%% calculate the partial derivatives
dTheta = ones(2*radius +1,2*radius +1, 4);
dTheta(:,:,1) = thetaIo / sqrt(2*pi) / sigma .* ...
    (exp(-(X-thetaX-0.5).^2 / (2*sigma^2)) - ...
     exp(-(X-thetaX+0.5).^2 / (2*sigma^2)) ) .* deltaEy;
dTheta(:,:,2) = thetaIo / sqrt(2*pi) / sigma .* ...
    (exp(-(Y-thetaY-0.5).^2 / (2*sigma^2)) - ...
     exp(-(Y-thetaY+0.5).^2 / (2*sigma^2)) ) .* deltaEx;
dTheta(:,:,3) = deltaEx .* deltaEy;
dTheta(:,:,4) = ones(size(MU));

%% calculate Fisher information matrix
I = zeros(4,4);
for i = 1:4
    for j = i:4
        I(i,j) = sum(sum(1./MU.*dTheta(:,:,i).*dTheta(:,:,j)));
        I(j,i) = I(i,j);
    end
end

%%
cr=sqrt(abs(inv(I)));
CR=diag(cr);
% end

