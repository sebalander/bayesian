# generates nData points out of a nPol grade polynomial in the interval 0-1.
# sNoise is the noise of the data

function [x, y] = generateData(nData,nPol,sNoise)
    
    x = rand(1,nData);
    
    roots = rand(1,nPol); 
    
    y = ones(1,nData);
    for i=1:nPol
        y = y .* (x - roots(i));
    endfor
    
    y = y + randn(1,nData)*sNoise;
    
endfunction