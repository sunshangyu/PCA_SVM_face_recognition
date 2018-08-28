%A为样本矩阵，k为降维维度数，mA为平均图片
function [pcaA, V] = fastPCA(A, k, mA)
m = size(A, 1);
%中心化样本矩阵
%repmat
%功能是以A的内容堆叠在（MxN）的矩阵B中，
%B矩阵的大小由MxN及A矩阵的内容决定，
%如果A是一个3x4x5的矩阵，有B = repmat(A,2,3)则最后的矩阵是6x12x5
Z = (A - repmat(mA, m, 1));
T = Z * Z';
%计算T的最大的k个特征值和特征向量
[V1, D] = eigs(T, k);
%协方差矩阵的特征向量
V = Z' * V1;
%特征向量单位化
for i = 1 : k
    %l是V(:, i)的最大奇异值
    l = norm(V(:, i));
    V(:, i) = V(:, i) / l;
end
%线性变换，降至k维，将中心化的矩阵投影到低维空间的基中
%V就是低维空间的基
pcaA = Z * V;
end