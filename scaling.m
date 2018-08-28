%特征数据规范化    
%将同一个样本中的不同维度归一化
%faceMat需要进行规范化的图像数据，    
%lowvec原来图像数据中的最小值    
%upvec原来图像数据中的最大值  
function [scaledface] = scaling(faceMat, lowvec, upvec)
upnew = 1;
lownew = -1;
[m, n] = size(faceMat);
scaledface = zeros(m, n);
for i = 1 : m
    scaledface(i, :) = lownew + (faceMat(i, :) - lowvec) ./ ...
        (upvec - lowvec) * (upnew - lownew);
end
end