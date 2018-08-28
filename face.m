clc;
clear;
close all;
disp('��ȡѵ������......');
%��ȡ15 * 7��ѵ��ͼƬ
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
        realclass((i - 1) * 7 + j) = i;%��ÿһ��ͼƬ��ӱ�ǩ
        all_stuff((i - 1) * 7 + j, :) = b(:)';
    end
end
%��ѵ�������н�ά����
disp('��ȡѵ������ƽ����......');
average_sample = mean(all_stuff);
%�洢ƽ����
disp('�洢ѵ������ƽ����......');
temp_average_sample = mat2gray(reshape(average_sample,147,137));
imshow(temp_average_sample);
imwrite(temp_average_sample, strcat('C:\Users\sun',...
        '\Desktop\SVM\ƽ����.jpg'));
%�����ĸ�����������ά��
disp('��ȡѵ������PCA����......');
k = 50;
[train_pcaface, V] = fastPCA(all_stuff, k, average_sample);
disp('�洢������......'); 
img = zeros(147, 137);
for i = 1 : 50
    img(:) = V(:, i);
    imwrite(im2uint8(mat2gray(img)), strcat('C:\Users\sun',...
        '\Desktop\SVM\Chara_face\',num2str(i),'.bmp'));
end
%��άѵ������һ��
disp('ѵ���������ݹ�һ��......');
lowvec = min(train_pcaface);
upvec = max(train_pcaface);
train_scaledface = scaling(train_pcaface, lowvec, upvec);
%SVM����ѵ��
disp('SVM����ѵ��......');
model = svmtrain(realclass, train_scaledface, '-t 0');
%��ȡ��������
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
		testclass((i - 1) * 4 + j) = i;%��ÿһ��ͼƬ��ӱ�ǩ
		test_stuff((i - 1) * 4 + j, :) = b(:)';
	end
end
disp('�������ݽ�ά......');
for i = 1 : 60
	test_stuff(i, :) = test_stuff(i, :) - average_sample;
end  
test_pcatestface = test_stuff * V;
%�������ݹ�һ��
disp('�������ݹ�һ��......');
scaled_testface = scaling(test_pcatestface, lowvec, upvec);
%����ѵ����������ģ�ͣ��Բ��Լ����з���
disp('SVM��������......');
[predict_label, accuracy, decision_values] = svmpredict(testclass, scaled_testface, model);
disp('ʶ����:');
fprintf('׼ȷ�� %.2f\n',accuracy(1,1) * 100);