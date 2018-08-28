clc;
clear;
close all;
disp('读取训练数据......');
%读取15 * 7张训练图片
realclass = zeros(15 * 7, 1);
all_stuff = zeros(15 * 7, 137 * 147);
for i = 1 : 15
    for j = 1 : 7
        if(i < 10)
            a = imread(strcat('C:\Users\sun\Desktop\SVM\face_store\00',...
                num2str(i),'\\0',num2str(j),'.jpg'));
        else
            a = imread(strcat('C:\Users\sun\Desktop\SVM\face_store\0',...
                num2str(i),'\\0',num2str(j),'.jpg'));
        end
        b = a(1 : 137 * 147);b = double(b);
        realclass((i - 1) * 7 + j) = i;%给每一张图片添加标签
        all_stuff((i - 1) * 7 + j, :) = b(:)';
    end
end
%对训练集进行降维处理
disp('提取训练样本平均脸......');
average_sample = mean(all_stuff);
%存储平均脸
disp('存储训练样本平均脸......');
temp_average_sample = mat2gray(reshape(average_sample,147,137));
imshow(temp_average_sample);
imwrite(temp_average_sample, strcat('C:\Users\sun',...
        '\Desktop\SVM\平均脸.jpg'));
%特征的个数，降至的维度
disp('提取训练样本PCA特征......');
k = 50;
[train_pcaface, V] = fastPCA(all_stuff, k, average_sample);
disp('存储特征脸......'); 
img = zeros(147, 137);
for i = 1 : 50
    img(:) = V(:, i);
    imwrite(im2uint8(mat2gray(img)), strcat('C:\Users\sun',...
        '\Desktop\SVM\Chara_face\',num2str(i),'.bmp'));
end
%低维训练集归一化
disp('训练特征数据归一化......');
lowvec = min(train_pcaface);
upvec = max(train_pcaface);
train_scaledface = scaling(train_pcaface, lowvec, upvec);
%SVM样本训练
disp('SVM样本训练......');
model = svmtrain(realclass, train_scaledface, '-t 0');
%读取测试数据
testclass = zeros(15 * 4, 1);
test_stuff = zeros(15 * 4, 137 * 147);
for i = 1 : 15
    for j = 8 : 11  
        if(i < 10)
            if(j < 10)
                a = imread(strcat('C:\Users\sun\Desktop\SVM\face_store\00',...
                num2str(i),'\0',num2str(j),'.jpg'));
            else
                a = imread(strcat('C:\Users\sun\Desktop\SVM\face_store\00',...
                num2str(i),'\\',num2str(j),'.jpg'));
            end
        else
            if(j < 10)
                a = imread(strcat('C:\Users\sun\Desktop\SVM\face_store\0',...
                num2str(i),'\\0',num2str(j),'.jpg'));
            else
                a = imread(strcat('C:\Users\sun\Desktop\SVM\face_store\0',...
                num2str(i),'\\',num2str(j),'.jpg'));
            end
        end
        b = a(1 : 137 * 147);b = double(b);
		testclass((i - 1) * 4 + j) = i;%给每一张图片添加标签
		test_stuff((i - 1) * 4 + j, :) = b(:)';
	end
end
disp('测试数据降维......');
for i = 1 : 60
	test_stuff(i, :) = test_stuff(i, :) - average_sample;
end  
test_pcatestface = test_stuff * V;
%测试数据归一化
disp('测试数据归一化......');
scaled_testface = scaling(test_pcatestface, lowvec, upvec);
%利用训练集建立的模型，对测试集进行分类
disp('SVM样本分类......');
[predict_label, accuracy, decision_values] = svmpredict(testclass, scaled_testface, model);
disp('识别率:');
fprintf('准确率 %.2f\n',accuracy(1,1) * 100);