%�������ݹ淶��    
%��ͬһ�������еĲ�ͬά�ȹ�һ��
%faceMat��Ҫ���й淶����ͼ�����ݣ�    
%lowvecԭ��ͼ�������е���Сֵ    
%upvecԭ��ͼ�������е����ֵ  
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